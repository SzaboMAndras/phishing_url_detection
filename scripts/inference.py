from predict import predict

# 1. Default (Improved Model)
res_imp = predict("https://www.PayPal-secure.login.tk/webscr?cmd=x")
print("Improved Model:", res_imp)

# 2. Baseline Model
res_base = predict(
    "https://www.PayPal-secure.login.tk/webscr?cmd=x", model_type="baseline"
)
print("Baseline Model:", res_base)