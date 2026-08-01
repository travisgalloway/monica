// A minimal pure-Swift Parquet reader — just enough to pull the `text` column out of the
// shards `src/data/corpus.py::write_shards` produces, so `monica-tokenize pack` can read them
// directly instead of round-tripping through `cleaned.jsonl`. See the #247 plan for the full
// design rationale (dependency survey, encoding surface probed against real pyarrow output).
//
// Deliberately narrow: UNCOMPRESSED/SNAPPY only, DATA_PAGE V1 only, flat non-repeated schemas,
// BYTE_ARRAY columns only. Anything outside that is a named, catchable error — never a silent
// skip and never a wrong-input tokenization (same discipline as `PackingError`/`TokenizerError`).

import Foundation

public enum ParquetError: Error, CustomStringConvertible {
    case notParquet(String)
    case truncated(String)
    case malformed(String)
    case unsupportedCodec(codec: String, file: String)
    case unsupportedEncoding(encoding: String, file: String)
    case unsupportedPageType(page: String, file: String)
    case unsupportedSchema(String, file: String)
    case columnNotFound(column: String, file: String)

    public var description: String {
        switch self {
        case .notParquet(let m): return "not a Parquet file: \(m)"
        case .truncated(let m): return "truncated Parquet file: \(m)"
        case .malformed(let m): return "malformed Parquet file: \(m)"
        case .unsupportedCodec(let codec, let file):
            return "unsupported Parquet codec \(codec) in \(file); re-write the shards with " +
                   "compression=\"snappy\" (src/data/corpus.py write_shards)"
        case .unsupportedEncoding(let encoding, let file):
            return "unsupported Parquet encoding \(encoding) in \(file)"
        case .unsupportedPageType(let page, let file):
            return "unsupported Parquet page type \(page) in \(file) (only DATA_PAGE / " +
                   "DICTIONARY_PAGE V1 are supported)"
        case .unsupportedSchema(let m, let file):
            return "unsupported Parquet schema in \(file): \(m)"
        case .columnNotFound(let column, let file):
            return "column '\(column)' not found in \(file)"
        }
    }
}

public enum Parquet {
    /// Every value of `column` from `url`, in file order (row group order, then page order,
    /// then row order). Null values are skipped and reported in `nullsSkipped`; empty strings
    /// are kept. Throws `ParquetError` on anything outside the supported subset — never
    /// silently mis-decodes.
    public static func readStringColumn(contentsOf url: URL, column: String = "text")
        throws -> (values: [String], nullsSkipped: Int)
    {
        let fileName = url.lastPathComponent
        // Memory-mapped where possible so a 128 MB shard is not fully resident; we still only
        // ever materialize small `Array` slices (one column chunk / one page at a time) out of
        // it below — never `Array(data)` on the whole file.
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw ParquetError.truncated("cannot read \(fileName): \(error)")
        }
        let n = data.count
        guard n >= 12 else { throw ParquetError.notParquet("\(fileName) is only \(n) bytes") }

        // `Data` returned by `Data(contentsOf:)` may not start at index 0, and — the gotcha —
        // a `Data` *slice* keeps the parent's absolute indices (`slice[0]` traps unless it
        // happens to start at 0). We deal with that in exactly one place: every read below
        // that needs an `Array` re-indexes via `Array(data[base+lo..<base+hi])`, which
        // normalizes to 0-based indices as part of the conversion.
        let base = data.startIndex
        func byte(_ i: Int) -> UInt8 { data[base + i] }
        func magic(at: Int) -> [UInt8] { (0..<4).map { byte(at + $0) } }

        let par1 = Array("PAR1".utf8)
        guard magic(at: 0) == par1 else {
            throw ParquetError.notParquet("\(fileName) missing leading PAR1 magic")
        }
        let tail = magic(at: n - 4)
        guard tail == par1 else {
            if tail == Array("PARE".utf8) {
                throw ParquetError.notParquet(
                    "\(fileName) has an encrypted footer (PARE) — not supported")
            }
            throw ParquetError.notParquet("\(fileName) missing trailing PAR1 magic")
        }
        let footerLen = Int(UInt32(byte(n - 8)) | UInt32(byte(n - 7)) << 8
            | UInt32(byte(n - 6)) << 16 | UInt32(byte(n - 5)) << 24)
        guard 12 + footerLen <= n else {
            throw ParquetError.truncated(
                "\(fileName) footer length \(footerLen) exceeds file size \(n)")
        }
        let footerStart = n - 8 - footerLen
        let footerBytes = Array(data[(base + footerStart)..<(base + n - 8)])
        var footerReader = ThriftCompactReader(footerBytes)
        let meta: ParquetFileMetaData
        do { meta = try ParquetMetadataDecoder.decodeFileMetaData(&footerReader) }
        catch { throw ParquetError.malformed("\(fileName) footer: \(error)") }

        // Schema: element 0 is the root; require flat (its children are leaves, not groups).
        guard !meta.schema.isEmpty else {
            throw ParquetError.unsupportedSchema("empty schema", file: fileName)
        }
        let root = meta.schema[0]
        let children = Array(meta.schema.dropFirst())
        guard children.count == Int(root.numChildren) else {
            throw ParquetError.unsupportedSchema(
                "root num_children (\(root.numChildren)) does not match \(children.count) " +
                "trailing schema elements", file: fileName)
        }
        guard children.allSatisfy({ $0.numChildren == 0 }) else {
            throw ParquetError.unsupportedSchema("nested (non-flat) schema is not supported", file: fileName)
        }
        guard let col = children.first(where: { $0.name == column }) else {
            throw ParquetError.columnNotFound(column: column, file: fileName)
        }
        guard col.type == ParquetPhysicalType.byteArray else {
            throw ParquetError.unsupportedSchema(
                "column '\(column)' has physical type " +
                "\(ParquetPhysicalType.name(col.type ?? -1)), expected BYTE_ARRAY", file: fileName)
        }
        let repetitionType = col.repetitionType ?? ParquetRepetitionType.required
        guard repetitionType != ParquetRepetitionType.repeated else {
            throw ParquetError.unsupportedSchema(
                "column '\(column)' is REPEATED (lists are not supported)", file: fileName)
        }
        let maxDef = repetitionType == ParquetRepetitionType.optional ? 1 : 0

        var values: [String] = []
        var nullsSkipped = 0

        for rg in meta.rowGroups {
            guard let chunk = rg.columns.first(where: { $0.metaData?.pathInSchema == [column] }) else {
                throw ParquetError.columnNotFound(column: column, file: fileName)
            }
            guard let cmeta = chunk.metaData else {
                throw ParquetError.malformed("\(fileName): column chunk missing meta_data")
            }
            if let fp = chunk.filePath, !fp.isEmpty {
                throw ParquetError.unsupportedSchema(
                    "column chunk stored in an external file (\(fp)) is not supported", file: fileName)
            }
            guard cmeta.codec == ParquetCodec.uncompressed || cmeta.codec == ParquetCodec.snappy else {
                throw ParquetError.unsupportedCodec(codec: ParquetCodec.name(cmeta.codec), file: fileName)
            }

            // `file_offset` is unreliable (always 0 in practice) — use the page offsets instead.
            let start = (cmeta.dictionaryPageOffset > 0 && cmeta.dictionaryPageOffset < cmeta.dataPageOffset)
                ? Int(cmeta.dictionaryPageOffset) : Int(cmeta.dataPageOffset)
            let end = start + Int(cmeta.totalCompressedSize)
            guard start >= 0, start <= end, end <= n else {
                throw ParquetError.truncated(
                    "\(fileName): column chunk range [\(start), \(end)) out of bounds (file is \(n) bytes)")
            }
            let chunkBytes = Array(data[(base + start)..<(base + end)])

            var dictionary: [[UInt8]] = []   // each row group has its own — reset here
            var localOff = 0
            var valuesRead = 0
            let numValues = Int(cmeta.numValues)

            while localOff < chunkBytes.count && valuesRead < numValues {
                var hr = ThriftCompactReader(chunkBytes, at: localOff)
                let ph: ParquetPageHeader
                do { ph = try ParquetMetadataDecoder.decodePageHeader(&hr) }
                catch { throw ParquetError.malformed("\(fileName): page header: \(error)") }
                let bodyStart = hr.pos
                let compLen = Int(ph.compressedPageSize)
                let uncompLen = Int(ph.uncompressedPageSize)
                guard compLen >= 0, uncompLen >= 0, bodyStart + compLen <= chunkBytes.count else {
                    throw ParquetError.truncated("\(fileName): page body out of bounds")
                }
                let rawBody = Array(chunkBytes[bodyStart..<(bodyStart + compLen)])
                let body: [UInt8]
                switch cmeta.codec {
                case ParquetCodec.uncompressed:
                    guard rawBody.count == uncompLen else {
                        throw ParquetError.malformed(
                            "\(fileName): uncompressed page size mismatch (\(rawBody.count) != \(uncompLen))")
                    }
                    body = rawBody
                default:   // SNAPPY — the only other codec accepted above
                    do { body = try Snappy.decompress(rawBody) }
                    catch { throw ParquetError.malformed("\(fileName): \(error)") }
                    guard body.count == uncompLen else {
                        throw ParquetError.malformed(
                            "\(fileName): decompressed page size mismatch (\(body.count) != \(uncompLen))")
                    }
                }

                switch ph.type {
                case ParquetPageType.dictionaryPage:
                    guard let dph = ph.dictionaryPageHeader else {
                        throw ParquetError.malformed("\(fileName): DICTIONARY_PAGE missing dictionary_page_header")
                    }
                    guard dph.encoding == ParquetEncoding.plain || dph.encoding == ParquetEncoding.plainDictionary else {
                        throw ParquetError.unsupportedEncoding(
                            encoding: ParquetEncoding.name(dph.encoding) + " (dictionary page)", file: fileName)
                    }
                    dictionary = try decodeDictionaryPage(body, count: Int(dph.numValues), file: fileName)

                case ParquetPageType.dataPage:
                    guard let dh = ph.dataPageHeader else {
                        throw ParquetError.malformed("\(fileName): DATA_PAGE missing data_page_header")
                    }
                    let pageValues = try decodeDataPageV1(
                        body, header: dh, maxDef: maxDef, dictionary: dictionary, file: fileName)
                    for v in pageValues {
                        if let bytes = v { values.append(String(decoding: bytes, as: UTF8.self)) }
                        else { nullsSkipped += 1 }
                    }
                    valuesRead += pageValues.count

                default:   // DATA_PAGE_V2, INDEX_PAGE — deliberately rejected, not a silent skip
                    throw ParquetError.unsupportedPageType(page: ParquetPageType.name(ph.type), file: fileName)
                }

                localOff = bodyStart + compLen
            }

            guard valuesRead == numValues else {
                throw ParquetError.malformed(
                    "\(fileName): row group column chunk expected \(numValues) values, read \(valuesRead)")
            }
        }

        return (values, nullsSkipped)
    }

    // MARK: - page bodies

    private static func decodeDictionaryPage(_ body: [UInt8], count: Int, file: String) throws -> [[UInt8]] {
        var dict: [[UInt8]] = []
        dict.reserveCapacity(count)
        var pos = 0
        for _ in 0..<count {
            guard pos + 4 <= body.count else {
                throw ParquetError.truncated("\(file): dictionary page entry length truncated")
            }
            let len = Int(UInt32(body[pos]) | UInt32(body[pos + 1]) << 8
                | UInt32(body[pos + 2]) << 16 | UInt32(body[pos + 3]) << 24)
            pos += 4
            guard pos + len <= body.count else {
                throw ParquetError.truncated("\(file): dictionary page entry bytes truncated")
            }
            dict.append(Array(body[pos..<(pos + len)]))
            pos += len
        }
        return dict
    }

    /// Returns one entry per row in the page (`nil` = null), in row order.
    private static func decodeDataPageV1(_ body: [UInt8], header: ParquetDataPageHeader, maxDef: Int,
                                         dictionary: [[UInt8]], file: String) throws -> [[UInt8]?] {
        let numValues = Int(header.numValues)
        var pos = 0
        // REQUIRED (maxDef == 0): no definition-level section at all; every row is non-null.
        var defLevels = Array(repeating: maxDef, count: numValues)

        if maxDef > 0 {
            guard header.definitionLevelEncoding == ParquetEncoding.rle else {
                throw ParquetError.unsupportedEncoding(
                    encoding: ParquetEncoding.name(header.definitionLevelEncoding) + " (definition levels; " +
                              "BIT_PACKED is the deprecated form and is not supported)", file: file)
            }
            guard pos + 4 <= body.count else { throw ParquetError.truncated("\(file): definition-level length") }
            let byteLen = Int(UInt32(body[pos]) | UInt32(body[pos + 1]) << 8
                | UInt32(body[pos + 2]) << 16 | UInt32(body[pos + 3]) << 24)
            pos += 4
            guard pos + byteLen <= body.count else {
                throw ParquetError.truncated("\(file): definition levels truncated")
            }
            let levelBytes = Array(body[pos..<(pos + byteLen)])
            pos += byteLen
            // maxDef is always exactly 1 here (REPEATED is rejected earlier, REQUIRED takes the
            // `maxDef == 0` branch above) → the level bit width is always 1.
            do { defLevels = try RleHybrid.decode(levelBytes, count: numValues, bitWidth: 1) }
            catch { throw ParquetError.malformed("\(file): definition levels: \(error)") }
        }

        let nonNullCount = defLevels.reduce(0) { $0 + ($1 == maxDef ? 1 : 0) }
        let valuesSection = pos <= body.count ? Array(body[pos...]) : []

        var rawValues: [[UInt8]] = []
        rawValues.reserveCapacity(nonNullCount)
        switch header.encoding {
        case ParquetEncoding.rleDictionary, ParquetEncoding.plainDictionary:
            if nonNullCount > 0 {
                guard !valuesSection.isEmpty else {
                    throw ParquetError.truncated("\(file): dictionary-index section missing bit-width byte")
                }
                let bitWidth = Int(valuesSection[0])
                let idxBytes = Array(valuesSection.dropFirst())
                let idxs: [Int]
                do { idxs = try RleHybrid.decode(idxBytes, count: nonNullCount, bitWidth: bitWidth) }
                catch { throw ParquetError.malformed("\(file): dictionary indices: \(error)") }
                for idx in idxs {
                    guard idx >= 0 && idx < dictionary.count else {
                        throw ParquetError.malformed(
                            "\(file): dictionary index \(idx) out of range [0, \(dictionary.count))")
                    }
                    rawValues.append(dictionary[idx])
                }
            }
        case ParquetEncoding.plain:
            var p = 0
            for _ in 0..<nonNullCount {
                guard p + 4 <= valuesSection.count else {
                    throw ParquetError.truncated("\(file): PLAIN value length truncated")
                }
                let len = Int(UInt32(valuesSection[p]) | UInt32(valuesSection[p + 1]) << 8
                    | UInt32(valuesSection[p + 2]) << 16 | UInt32(valuesSection[p + 3]) << 24)
                p += 4
                guard p + len <= valuesSection.count else {
                    throw ParquetError.truncated("\(file): PLAIN value bytes truncated")
                }
                rawValues.append(Array(valuesSection[p..<(p + len)]))
                p += len
            }
        default:
            throw ParquetError.unsupportedEncoding(encoding: ParquetEncoding.name(header.encoding), file: file)
        }

        guard rawValues.count == nonNullCount else {
            throw ParquetError.malformed(
                "\(file): expected \(nonNullCount) non-null values, decoded \(rawValues.count)")
        }

        var out: [[UInt8]?] = []
        out.reserveCapacity(numValues)
        var vi = 0
        for d in defLevels {
            if d == maxDef { out.append(rawValues[vi]); vi += 1 } else { out.append(nil) }
        }
        return out
    }
}
