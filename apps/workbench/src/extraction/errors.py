"""文件读取模块异常。"""


class ExtractionError(Exception):
    """文件读取基础异常。"""


class UnsupportedFileTypeError(ExtractionError):
    """输入文件格式不在支持范围内。"""
