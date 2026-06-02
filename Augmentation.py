import os
import sys
import numpy as np
import cv2 # Lib ComputerVision OpenCv
import matplotlib.pyplot as plt


AUGMENT_TAGS = ["_Flip", "_Rotate", "_Shear", "_Crop", "_Contrast", "_Blur"]



def flip(img):
	# [param / Options]
	#	1 : horizontal -> miroir gauche / droite
	#	0 : vertical -> miroir haut / bas
	#	-1: les 2 -> rotation 180
	return cv2.flip(img, 1)



def rotate(img):
	h, w = img.shape[:2]
	center = (w // 2, h // 2)
	angle = 45
	matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

	return cv2.warpAffine(img, matrix, (w, h))



def skew(img):
	h, w = img.shape[:2]
	pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
	pts2 = np.float32([[w * 0.1, 0], [w, 0], [0, h], [w * 0.9, h]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)

	return cv2.warpPerspective(img, matrix, (w, h))



def shear(img):
	h, w = img.shape[:2]
	shear_factor = 0.3
	matrix = np.float32(
		[
			[1, shear_factor, 0],
			[0, 1, 0]
		]
	)
	new_w = int(w + shear_factor * h)

	return cv2.warpAffine(img, matrix, (new_w, h))



def crop(img):
	h, w = img.shape[:2]
	margin_x = int(w * 0.1)
	margin_y = int(h * 0.1)
	cropped = img[margin_y:h - margin_y, margin_x:w - margin_x]

	return cv2.resize(cropped, (w, h))



def distortion(img):
	h, w = img.shape[:2]
	map_x = np.zeros((h, w), np.float32)
	map_y = np.zeros((h, w), np.float32)

	for i in range(h):
		for j in range(w):
			map_x[i, j] = j + 10 * np.sin(i / 20)
			map_y[i, j] = i + 10 * np.sin(j / 20)

	return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)



def contrast(img):
	alpha = 1.5  # contrastes (1= original,  > 1 = augmente contrastes)
	beta = 0     # luminosite (0 = pas de changement)

	return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)



def blur(img):

	return cv2.GaussianBlur(img, (15, 15), 0)



def illumination(img):
	h, w = img.shape[:2]
	x, y = np.meshgrid(np.arange(w), np.arange(h))
	dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
	mask = np.clip(1 - dist / (min(h, w) // 2), 0, 1).astype(np.float32)
	illuminated = img.astype(np.float32)

	for c in range(3):
		illuminated[:, :, c] *= (1 + mask * 0.8)

	return np.clip(illuminated, 0, 255).astype(np.uint8)



def save_augmented(img, original_path, transform_name):
	base, ext = os.path.splitext(original_path)
	output_path = f"{base}_{transform_name}{ext}"
	cv2.imwrite(output_path, img)
	return output_path



def augment_directory(dir_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    augment_tags = AUGMENT_TAGS

    files = [
        f for f in os.listdir(dir_path)
        if f.endswith(valid_extensions)
        and not any(tag in f for tag in augment_tags)
    ]

    total = len(files)
    if total == 0:
        print(f"[ Error ]: no images found in '{dir_path}'")
        return

    print(f"\n🌿 Augmenting '{os.path.basename(dir_path)}' — {total} images found\n")

    for i, filename in enumerate(files, 1):
        img_path = os.path.join(dir_path, filename)
        augment_image(img_path)
        print(f"  [{i:>4}/{total}] ✅ {filename}")

    print(f"\n✅ Done — {total * len(augment_tags)} images generated\n")




def augment_image(img_path, display=False):

	img = cv2.imread(img_path)
	if img is None:
		print(f"[ Error ]: cannot read '{img_path}'")
		sys.exit(1)

	# save_augmented(img, img_path, "Original") # Sauvegarde l'Originale en plus !

	# We only select 6 Transformation Methods
	transformations = {
		"Original":   img,
		"Flip":       flip(img),
		"Rotate":     rotate(img),
		# "Skew":       skew(img),
		"Shear":      shear(img),
		"Crop":       crop(img),
		# "Distortion": distortion(img),
		"Contrast":   contrast(img),
		"Blur": blur(img),
		# "Illumniation": illumination(img),
	}

	if display:
		# Afficher toutes les transformations dans une grille
		n = len(transformations)
		cols = 4
		rows = (n + cols - 1) // cols  # arrondi supérieur
		fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
		axes = axes.flatten()

		for i, (name, transformed) in enumerate(transformations.items()):
			axes[i].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
			axes[i].set_title(name)
			axes[i].axis('off')

		# Cacher les axes vides si nombre impair
		for j in range(i + 1, len(axes)):
			axes[j].axis('off')

		fig.suptitle(os.path.basename(img_path), fontsize=12)
		plt.tight_layout()
		plt.show()

	else:
		for name, transformed in transformations.items():
			if name == "Original":    # ← skip l'original en mode save
				continue
			output_path = save_augmented(transformed, img_path, name)
			print(f"  ✅ saved: {output_path}")
#





def main():

	if len(sys.argv) != 2:
		print("usage: python3 Augmentation.py <image_path or directory_path>")
		return

	path = sys.argv[1]
	
	if os.path.isfile(path):
		augment_image(path, display=True)
	
	elif os.path.isdir(path):
		augment_directory(path)
	
	else:
		print(f"[ Error ]: '{path}' is not a valid file or directory Bro !")


	return



if __name__=="__main__":
	main()
