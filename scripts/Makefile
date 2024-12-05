
adult:
	TF_USE_LEGACY_KERAS=1 python train_adult.py --model ffn --format h5
	TF_USE_LEGACY_KERAS=1 python train_adult.py --model ffn
	python train_adult.py --model ffn --format keras


iris_ffn:
	TF_USE_LEGACY_KERAS=1 python train_iris.py --model ffn --format h5
	TF_USE_LEGACY_KERAS=1 python train_iris.py --model ffn
	python train_adult.py --model ffn --format keras


iris_ae:
	TF_USE_LEGACY_KERAS=1 python train_iris.py --model ae --format h5
	TF_USE_LEGACY_KERAS=1 python train_iris.py --model ae
	python train_iris.py --model ae --format keras


iris: iris_ffn iris_ae


mnist:
	TF_USE_LEGACY_KERAS=1 python train_mnist.py --model logistic --format h5
	TF_USE_LEGACY_KERAS=1 python train_mnist.py --model logistic
	python train_mnist.py --model logistic --format keras
	TF_USE_LEGACY_KERAS=1 python train_mnist.py --model cnn --format h5
	TF_USE_LEGACY_KERAS=1 python train_mnist.py --model cnn
	python train_mnist.py --model cnn --format keras
	python train_mnist_pt.py


all: adult iris mnist


clean:
	rm -rf datasets