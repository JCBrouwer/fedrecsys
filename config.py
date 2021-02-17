MODEL_NAME = "mnist"
MODEL_VERSION = "1.2.3"
GRID_ADDRESS = "127.0.0.1:5050"

# ### Authentication (optional)
# Let's additionally protect the model with simple authentication for workers.
#
# PyGrid supports authentication via JWT token (HMAC, RSA) or opaque token
# via remote API.
#
# We'll try JWT/RSA. Suppose we generate RSA keys:
# ```
# openssl genrsa -out private.pem
# openssl rsa -in private.pem -pubout -out public.pem
# ```

PRIVATE_KEY = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAzQMcI09qonB9OZT20X3Z/oigSmybR2xfBQ1YJ1oSjQ3YgV+G
FUuhEsGDgqt0rok9BreT4toHqniFixddncTHg7EJzU79KZelk2m9I2sEsKUqEsEF
lMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYvGFphwwh4TNJXxkCg69/RsvPBIPi2
9vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNVQhUFABDyWN4h/67M1eArGA540vyd
kYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+LzmjEnjTJqUzr7kM9Rzq3BY01DNi
TVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3ZQIDAQABAoIBAD+xbKeHv+BxxGYE
Yt5ZFEYhGnOk5GU/RRIjwDSRplvOZmpjTBwHoCZcmsgZDqo/FwekNzzuch1DTnIV
M0+V2EqQ0TPJC5xFcfqnikybrhxXZAfpkhtU+gR5lDb5Q+8mkhPAYZdNioG6PGPS
oGz8BsuxINhgJEfxvbVpVNWTdun6hLOAMZaH3DHgi0uyTBg8ofARoZP5RIbHwW+D
p+5vd9x/x7tByu76nd2UbMp3yqomlB5jQktqyilexCIknEnfb3i/9jqFv8qVE5P6
e3jdYoJY+FoomWhqEvtfPpmUFTY5lx4EERCb1qhWG3a7sVBqTwO6jJJBsxy3RLIS
Ic0qZcECgYEA6GsBP11a2T4InZ7cixd5qwSeznOFCzfDVvVNI8KUw+n4DOPndpao
TUskWOpoV8MyiEGdQHgmTOgGaCXN7bC0ERembK0J64FI3TdKKg0v5nKa7xHb7Qcv
t9ccrDZVn4y/Yk5PCqjNWTR3/wDR88XouzIGaWkGlili5IJqdLEvPvUCgYEA4dA+
5MNEQmNFezyWs//FS6G3lTRWgjlWg2E6BXXvkEag6G5SBD31v3q9JIjs+sYdOmwj
kfkQrxEtbs173xgYWzcDG1FI796LTlJ/YzuoKZml8vEF3T8C4Bkbl6qj9DZljb2j
ehjTv5jA256sSUEqOa/mtNFUbFlBjgOZh3TCsLECgYAc701tdRLdXuK1tNRiIJ8O
Enou26Thm6SfC9T5sbzRkyxFdo4XbnQvgz5YL36kBnIhEoIgR5UFGBHMH4C+qbQR
OK+IchZ9ElBe8gYyrAedmgD96GxH2xAuxAIW0oDgZyZgd71RZ2iBRY322kRJJAdw
Xq77qo6eXTKpni7grjpijQKBgDHWRAs5DVeZkTwhoyEW0fRfPKUxZ+ZVwUI9sxCB
dt3guKKTtoY5JoOcEyJ9FdBC6TB7rV4KGiSJJf3OXAhgyP9YpNbimbZW52fhzTuZ
bwO/ZWC40RKDVZ8f63cNsiGz37XopKvNzu36SJYv7tY8C5WvvLsrd/ZxvIYbRUcf
/dgBAoGBAMdR5DXBcOWk3+KyEHXw2qwWcGXyzxtca5SRNLPR2uXvrBYXbhFB/PVj
h3rGBsiZbnIvSnSIE+8fFe6MshTl2Qxzw+F2WV3OhhZLLtBnN5qqeSe9PdHLHm49
XDce6NV2D1mQLBe8648OI5CScQENuRGxF2/h9igeR4oRRsM1gzJN
-----END RSA PRIVATE KEY-----
""".strip()

PUBLIC_KEY = """
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzQMcI09qonB9OZT20X3Z
/oigSmybR2xfBQ1YJ1oSjQ3YgV+GFUuhEsGDgqt0rok9BreT4toHqniFixddncTH
g7EJzU79KZelk2m9I2sEsKUqEsEFlMpkk9qkPHhJB5AQoClOijee7UNOF4yu3HYv
GFphwwh4TNJXxkCg69/RsvPBIPi29vXFQzFE7cbN6jSxiCtVrpt/w06jJUsEYgNV
QhUFABDyWN4h/67M1eArGA540vydkYdSIEQdknKHjPW62n4dvqDWxtnK0HyChsB+
LzmjEnjTJqUzr7kM9Rzq3BY01DNiTVcB2G8t/jICL+TegMGU08ANMKiDfSMGtpz3
ZQIDAQAB
-----END PUBLIC KEY-----
""".strip()
