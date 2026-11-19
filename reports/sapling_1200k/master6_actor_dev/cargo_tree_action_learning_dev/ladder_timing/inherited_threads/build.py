from pathlib import Path
import hashlib
H=Path(__file__).resolve().parent
P=H.parents[1]/'candidate.py'
data=P.read_bytes()
assert hashlib.sha256(data).hexdigest()=='8da4b465f7fa6e601e51bd7c2e81f25825cb59b9b5ab285f7f56d0f6bbfe1cac'
before=b'import numpy as np\n'
assert before in data[:300]
prefix, suffix=data.split(before,1)
(H/'candidate.py').write_bytes(prefix+b'os.environ["OPENBLAS_NUM_THREADS"] = "1"\n'+before+suffix)
