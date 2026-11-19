FROM python:3.12-slim
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    python -m pip install --no-cache-dir --retries 1 --timeout 15 --index-url https://pypi.org/simple 'numpy==1.26.4' 'requests==2.32.5' 'jsonschema==4.25.1'
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    python -m pip install --no-cache-dir --retries 1 --timeout 15 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple 'torch==2.5.1+cpu'
COPY code/env_src/carbon_challenge-master/zerosum_env /opt/official/zerosum_env
ENV PYTHONPATH=/opt/official
RUN python -c "from zerosum_env.envs.carbon.helpers import Board; import torch, numpy, sys; assert torch.__version__ == '2.5.1+cpu' and numpy.__version__ == '1.26.4' and sys.version_info[:2] == (3, 12); assert torch.tensor([2, 3]).sum().item() == 5"
