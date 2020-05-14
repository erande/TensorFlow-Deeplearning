import tensorflow as tf

# grad = (df_dw, df_db)

# y = xw + b
# dy_dw = x
# dy_db = 1
# grad = (x, 1)

# y = xw^2 + b^2
# dy_dw = x * 2w
# dy_db = 2b

# y = x *  e^w + e^b, (e^x)'=e^x
# dy_dw = x * e^w
# dy_db = e^b

# f = [y - (xw + b)]^2 => 单层感知机更新
# df_dw = 2 * (y - (xw + b)) * x
# df_db = 2 * (y - (xw + b))

# f = y * log(xw + b), (log x)' = 1 / x
# df_dw = xy / (xw + b)
# df_db = y / (xw + b)
