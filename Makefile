IMAGES_DIR="./images"
DB="./db"
IMAGE_TEST="./image.jpg"



init:
	sh ./init_env.sh



distribution:
	@(python3 ./Distribution.py $(IMAGES_DIR))

augmentation:
	@(python3 ./Augmentation.py $(IMAGE_TEST))

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



PHONY: init distribution augmentation transformation classification train predict freeze install clean fclean
