# Build from the repository root, after building the matching ladder_timing base.
FROM sapling-ladder-312:local
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' && python -m pip install --no-cache-dir 'requests==2.32.5' 'jsonschema==4.25.1'
COPY code/env_src/carbon_challenge-master/zerosum_env /opt/official/zerosum_env
ENV PYTHONPATH=/opt/official
RUN python -c "from zerosum_env.envs.carbon.helpers import Board; import torch, numpy, sys; assert torch.__version__ == '2.5.1+cpu' and numpy.__version__ == '1.26.4' and sys.version_info[:2] == (3, 12)"
