import tensorflow as tf

# grad = (df_dw, df_db)

# y = xw + b
# dy_dw = x
# dy_db = 1_start
# grad = (x, 1_start)

# y = xw^2 + b^2
# dy_dw = x * 2w
# dy_db = 2b

# y = x *  e^w + e^b, (e^x)'=e^x
# dy_dw = x * e^w
# dy_db = e^b

# f = [y - (xw + b)]^2 => 单层感知机更新
# df_dw = 2 * (y - (xw + b)) * x
# df_db = 2 * (y - (xw + b))

# f = y * log(xw + b), (log x)' = 1_start / x
# df_dw = xy / (xw + b)
# df_db = y / (xw + b)
