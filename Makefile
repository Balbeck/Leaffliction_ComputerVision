IMAGES_DIR="./images"
IMAGE_TEST="./images/Apple/Apple_Black_rot/image (1).JPG"
DB="./db"



init:
	sh ./init_env.sh



distribution:
	@(python3 ./Distribution.py $(IMAGES_DIR))


augmentation:
	@(python3 ./Augmentation.py $(IMAGE_TEST))

augApple:
	python3 ./Augmentation.py ./images/Apple/Apple_Black_rot
	python3 ./Augmentation.py ./images/Apple/Apple_healthy
	python3 ./Augmentation.py ./images/Apple/Apple_rust
	python3 ./Augmentation.py ./images/Apple/Apple_scab

augGrape:
	python3 ./Augmentation.py ./images/Grape/Grape_Black_rot
	python3 ./Augmentation.py ./images/Grape/Grape_healthy
	python3 ./Augmentation.py ./images/Grape/Grape_Esca
	python3 ./Augmentation.py ./images/Grape/Grape_spot

augAll: augApple augGrape 


transformation:
	@(python3 ./Transformation.py $(IMAGE_TEST))


train:
	@(rm -rf $(DB)_augmented $(DB)_transformed)
	@(python3 ./train.py $(DB))

predict:
	@(python3 ./predict.py --image_path $(IMAGE_TEST))



freeze:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt



clean:
	rm -rf */*/__pycache__

fclean: clean
	rm -rf .LeafflictionVenv



PHONY: init distribution augmentation augApple augGrape augAll transformation classification train predict freeze install clean fclean
