// Parquet FileMetaData / PageHeader — the subset ParquetReader.swift needs, decoded from the
// Thrift compact-protocol bytes in a file's footer and page headers. Field ids below are
// verified against real bytes written by `pyarrow` 24.0.0 (`corpus.py::write_shards`); every
// unlisted field is skipped, not decoded — see the field tables in the #247 plan.

import Foundation

// MARK: - enums (as plain constants; Thrift transmits these as i32)

/// `parquet.thrift` `Type` — physical column type.
enum ParquetPhysicalType {
    static let boolean: Int32 = 0
    static let int32: Int32 = 1
    static let int64: Int32 = 2
    static let int96: Int32 = 3
    static let float: Int32 = 4
    static let double: Int32 = 5
    static let byteArray: Int32 = 6
    static let fixedLenByteArray: Int32 = 7

    static func name(_ v: Int32) -> String {
        switch v {
        case boolean: return "BOOLEAN"
        case int32: return "INT32"
        case int64: return "INT64"
        case int96: return "INT96"
        case float: return "FLOAT"
        case double: return "DOUBLE"
        case byteArray: return "BYTE_ARRAY"
        case fixedLenByteArray: return "FIXED_LEN_BYTE_ARRAY"
        default: return "TYPE(\(v))"
        }
    }
}

/// `parquet.thrift` `FieldRepetitionType`.
enum ParquetRepetitionType {
    static let required: Int32 = 0
    static let optional: Int32 = 1
    static let repeated: Int32 = 2
}

/// `parquet.thrift` `PageType`.
enum ParquetPageType {
    static let dataPage: Int32 = 0
    static let indexPage: Int32 = 1
    static let dictionaryPage: Int32 = 2
    static let dataPageV2: Int32 = 3

    static func name(_ v: Int32) -> String {
        switch v {
        case dataPage: return "DATA_PAGE"
        case indexPage: return "INDEX_PAGE"
        case dictionaryPage: return "DICTIONARY_PAGE"
        case dataPageV2: return "DATA_PAGE_V2"
        default: return "PAGE_TYPE(\(v))"
        }
    }
}

/// `parquet.thrift` `Encoding`.
enum ParquetEncoding {
    static let plain: Int32 = 0
    static let plainDictionary: Int32 = 2
    static let rle: Int32 = 3
    static let bitPacked: Int32 = 4
    static let deltaBinaryPacked: Int32 = 5
    static let deltaLengthByteArray: Int32 = 6
    static let deltaByteArray: Int32 = 7
    static let rleDictionary: Int32 = 8
    static let byteStreamSplit: Int32 = 9

    static func name(_ v: Int32) -> String {
        switch v {
        case plain: return "PLAIN"
        case plainDictionary: return "PLAIN_DICTIONARY"
        case rle: return "RLE"
        case bitPacked: return "BIT_PACKED"
        case deltaBinaryPacked: return "DELTA_BINARY_PACKED"
        case deltaLengthByteArray: return "DELTA_LENGTH_BYTE_ARRAY"
        case deltaByteArray: return "DELTA_BYTE_ARRAY"
        case rleDictionary: return "RLE_DICTIONARY"
        case byteStreamSplit: return "BYTE_STREAM_SPLIT"
        default: return "ENCODING(\(v))"
        }
    }
}

/// `parquet.thrift` `CompressionCodec`.
enum ParquetCodec {
    static let uncompressed: Int32 = 0
    static let snappy: Int32 = 1
    static let gzip: Int32 = 2
    static let lzo: Int32 = 3
    static let brotli: Int32 = 4
    static let lz4: Int32 = 5
    static let zstd: Int32 = 6
    static let lz4Raw: Int32 = 7

    static func name(_ v: Int32) -> String {
        switch v {
        case uncompressed: return "UNCOMPRESSED"
        case snappy: return "SNAPPY"
        case gzip: return "GZIP"
        case lzo: return "LZO"
        case brotli: return "BROTLI"
        case lz4: return "LZ4"
        case zstd: return "ZSTD"
        case lz4Raw: return "LZ4_RAW"
        default: return "CODEC(\(v))"
        }
    }
}

// MARK: - structs

struct ParquetSchemaElement {
    var type: Int32? = nil               // absent for the root / group nodes
    var repetitionType: Int32? = nil     // absent for the root
    var name: String = ""
    var numChildren: Int32 = 0
}

struct ParquetColumnMetaData {
    var type: Int32 = 0
    var encodings: [Int32] = []
    var pathInSchema: [String] = []
    var codec: Int32 = ParquetCodec.uncompressed
    var numValues: Int64 = 0
    var totalCompressedSize: Int64 = 0
    var dataPageOffset: Int64 = 0
    /// `0` means absent (a real offset is always > 0 — past the 4-byte "PAR1" magic).
    var dictionaryPageOffset: Int64 = 0
}

struct ParquetColumnChunk {
    var filePath: String? = nil
    var fileOffset: Int64 = 0
    var metaData: ParquetColumnMetaData? = nil
}

struct ParquetRowGroup {
    var columns: [ParquetColumnChunk] = []
    var numRows: Int64 = 0
}

struct ParquetFileMetaData {
    var version: Int32 = 0
    var schema: [ParquetSchemaElement] = []
    var numRows: Int64 = 0
    var rowGroups: [ParquetRowGroup] = []
}

struct ParquetDataPageHeader {
    var numValues: Int32 = 0
    var encoding: Int32 = 0
    var definitionLevelEncoding: Int32 = 0
    var repetitionLevelEncoding: Int32 = 0
}

struct ParquetDictionaryPageHeader {
    var numValues: Int32 = 0
    var encoding: Int32 = 0
    var isSorted: Bool = false
}

struct ParquetPageHeader {
    var type: Int32 = 0
    var uncompressedPageSize: Int32 = 0
    var compressedPageSize: Int32 = 0
    var dataPageHeader: ParquetDataPageHeader? = nil
    var dictionaryPageHeader: ParquetDictionaryPageHeader? = nil
}

// MARK: - decoders
//
// Each function owns one Thrift struct and a *local* `lastFieldId`, which is what makes field
// numbering "per-struct" — see ThriftCompactReader's doc comment. Unrecognized field ids fall
// through to `reader.skip(type:)`.

enum ParquetMetadataDecoder {

    static func decodeFileMetaData(_ r: inout ThriftCompactReader) throws -> ParquetFileMetaData {
        var out = ParquetFileMetaData()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.version = try r.readZigzag32()
            case 2:
                let (size, elemType) = try r.readListHeader()
                guard elemType == ThriftType.structType else {
                    throw ThriftError.malformed("FileMetaData.schema: expected struct elements")
                }
                out.schema = try (0..<size).map { _ in try decodeSchemaElement(&r) }
            case 3: out.numRows = try r.readZigzag64()
            case 4:
                let (size, elemType) = try r.readListHeader()
                guard elemType == ThriftType.structType else {
                    throw ThriftError.malformed("FileMetaData.row_groups: expected struct elements")
                }
                out.rowGroups = try (0..<size).map { _ in try decodeRowGroup(&r) }
            default:
                try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodeSchemaElement(_ r: inout ThriftCompactReader) throws -> ParquetSchemaElement {
        var out = ParquetSchemaElement()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.type = try r.readZigzag32()
            case 3: out.repetitionType = try r.readZigzag32()
            case 4: out.name = try r.readString()
            case 5: out.numChildren = try r.readZigzag32()
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodeRowGroup(_ r: inout ThriftCompactReader) throws -> ParquetRowGroup {
        var out = ParquetRowGroup()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1:
                let (size, elemType) = try r.readListHeader()
                guard elemType == ThriftType.structType else {
                    throw ThriftError.malformed("RowGroup.columns: expected struct elements")
                }
                out.columns = try (0..<size).map { _ in try decodeColumnChunk(&r) }
            case 3: out.numRows = try r.readZigzag64()
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodeColumnChunk(_ r: inout ThriftCompactReader) throws -> ParquetColumnChunk {
        var out = ParquetColumnChunk()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.filePath = try r.readString()
            case 2: out.fileOffset = try r.readZigzag64()
            case 3: out.metaData = try decodeColumnMetaData(&r)
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodeColumnMetaData(_ r: inout ThriftCompactReader) throws -> ParquetColumnMetaData {
        var out = ParquetColumnMetaData()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.type = try r.readZigzag32()
            case 2:
                let (size, elemType) = try r.readListHeader()
                out.encodings = try (0..<size).map { _ in Int32(truncatingIfNeeded: try r.readIntElement(elemType)) }
            case 3:
                let (size, elemType) = try r.readListHeader()
                guard elemType == ThriftType.binary else {
                    throw ThriftError.malformed("ColumnMetaData.path_in_schema: expected binary elements")
                }
                out.pathInSchema = try (0..<size).map { _ in try r.readString() }
            case 4: out.codec = try r.readZigzag32()
            case 5: out.numValues = try r.readZigzag64()
            case 7: out.totalCompressedSize = try r.readZigzag64()
            case 9: out.dataPageOffset = try r.readZigzag64()
            case 11: out.dictionaryPageOffset = try r.readZigzag64()
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodePageHeader(_ r: inout ThriftCompactReader) throws -> ParquetPageHeader {
        var out = ParquetPageHeader()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.type = try r.readZigzag32()
            case 2: out.uncompressedPageSize = try r.readZigzag32()
            case 3: out.compressedPageSize = try r.readZigzag32()
            case 5: out.dataPageHeader = try decodeDataPageHeader(&r)
            case 7: out.dictionaryPageHeader = try decodeDictionaryPageHeader(&r)
            default: try r.skip(type: fh.type)   // includes field 8 data_page_header_v2 — V2 is
                                                  // rejected by PageType before we'd need it
            }
        }
        return out
    }

    static func decodeDataPageHeader(_ r: inout ThriftCompactReader) throws -> ParquetDataPageHeader {
        var out = ParquetDataPageHeader()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.numValues = try r.readZigzag32()
            case 2: out.encoding = try r.readZigzag32()
            case 3: out.definitionLevelEncoding = try r.readZigzag32()
            case 4: out.repetitionLevelEncoding = try r.readZigzag32()
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }

    static func decodeDictionaryPageHeader(_ r: inout ThriftCompactReader) throws -> ParquetDictionaryPageHeader {
        var out = ParquetDictionaryPageHeader()
        var lastFieldId: Int16 = 0
        while let fh = try r.readFieldBegin(lastFieldId: &lastFieldId) {
            switch fh.id {
            case 1: out.numValues = try r.readZigzag32()
            case 2: out.encoding = try r.readZigzag32()
            case 3: out.isSorted = (fh.type == ThriftType.boolTrue)   // value is in the type nibble
            default: try r.skip(type: fh.type)
            }
        }
        return out
    }
}
