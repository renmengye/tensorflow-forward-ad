cd tensorflow_forward_ad
python setup_cbfs.py build_ext --inplace
cd ..
VERBOSE=-1 TF_CPP_MIN_LOG_LEVEL=2 python -m unittest discover -v -s . -p "*_tests.py"
