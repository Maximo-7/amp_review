FROM python:3.7-bullseye

RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip==22.2.2 \
    wheel==0.32.1 \
    setuptools==49.1.0 \
    Cython==0.29.32

RUN pip install --no-cache-dir \
    https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.0/MindSpore/cpu/x86_64/mindspore-1.8.0-cp37-cp37m-linux_x86_64.whl

RUN pip install --no-cache-dir \
    addict==2.4.0 \
    asttokens==2.0.8 \
    astunparse==1.6.3 \
    attrs==22.1.0 \
    certifi==2022.6.15 \
    cffi==1.12.3 \
    charset-normalizer==2.1.1 \
    click==8.1.3 \
    cycler==0.11.0 \
    decorator==5.1.1 \
    easydict==1.9 \
    Flask==2.2.2 \
    future==0.18.2 \
    google-pasta==0.2.0 \
    grpcio==1.47.0 \
    grpcio-tools==1.47.0 \
    gunicorn==20.1.0 \
    idna==3.3 \
    importlib-metadata==4.12.0 \
    itsdangerous==2.1.2 \
    Jinja2==3.1.2 \
    joblib==1.1.0 \
    kiwisolver==1.4.4 \
    lxml==4.5.2 \
    MarkupSafe==2.1.1 \
    marshmallow==3.17.1 \
    matplotlib==3.2.2 \
    mmcv==0.2.14 \
    mpmath==1.2.1 \
    numexpr==2.8.3 \
    numpy==1.21.6 \
    opencv-python==4.5.5.64 \
    packaging==21.3 \
    pandas==1.0.5 \
    pathlib2==2.3.7.post1 \
    Pillow==9.2.0 \
    pip==22.2.2 \
    protobuf==3.20.1 \
    psutil==5.9.1 \
    pycocotools==2.0.1 \
    pycparser==2.21 \
    pyparsing==3.0.9 \
    python-dateutil==2.8.2 \
    pytz==2022.2.1 \
    PyYAML==6.0 \
    requests==2.28.1 \
    scikit-learn==0.24.1 \
    scipy==1.7.3 \
    six==1.16.0 \
    sklearn==0.0 \
    sympy==1.4 \
    tables==3.6.1 \
    threadpoolctl==3.1.0 \
    tqdm==4.66.4 \
    treelib==1.6.1 \
    typing_extensions==4.3.0 \
    urllib3==1.26.12 \
    Werkzeug==2.2.2 \
    XlsxWriter==3.0.3 \
    yapf==0.32.0 \
    zipp==3.8.1

RUN git clone https://github.com/hgao12345/DLFea4AMPGen.git /app/dlfea4ampgen \
    && cd /app/dlfea4ampgen \
    && git checkout 85e8c2e

WORKDIR /app/dlfea4ampgen

# Copy pretrained model(s)
COPY models/dlfea4ampgen/ABP_Best_Model.ckpt /app/dlfea4ampgen/models/ABP_Best_Model.ckpt
