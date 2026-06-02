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

transDir:
	@(python3 Transformation.py -src $(IMAGES_DIR)/Apple/apple_healthy/ -dst ./dst/ -mask)

transApple:
	python3 ./Transformation.py -src ./images/Apple/Apple_Black_rot -dst ./dst/Apple/
	python3 ./Transformation.py -src ./images/Apple/Apple_healthy -dst ./dst/Apple/
	python3 ./Transformation.py -src ./images/Apple/Apple_rust -dst ./dst/Apple/
	python3 ./Transformation.py -src ./images/Apple/Apple_scab -dst ./dst/Apple/

transGrape:
	python3 ./Transformation.py -src ./images/Grape/Grape_Black_rot -dst ./dst/Grape/
	python3 ./Transformation.py -src ./images/Grape/Grape_healthy -dst ./dst/Grape/
	python3 ./Transformation.py -src ./images/Grape/Grape_Esca -dst ./dst/Grape/
	python3 ./Transformation.py -src ./images/Grape/Grape_spot -dst ./dst/Grape/

transAll: transApple transGrape



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



PHONY: init distribution augmentation augApple augGrape augAll transformation transDir transApple transGrape transAll classification train predict freeze install clean fclean
