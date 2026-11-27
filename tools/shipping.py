"""示例业务工具，金额单位为元。"""


def shipping_fee(subtotal_yuan: int) -> str:
    """根据非负商品小计（元）查询运费，满 99 元免运费。"""
    if subtotal_yuan < 0:
        raise ValueError("商品小计必须大于或等于 0")
    return "0 元" if subtotal_yuan >= 99 else "8 元"
