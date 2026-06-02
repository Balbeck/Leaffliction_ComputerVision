import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt



def gaussian_blur(img):
	return cv2.GaussianBlur(img, (5, 5), 0)



def mask(img):
    # Convertir en LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Canal A : vert (-) vs rouge (+)
    a_channel = lab[:, :, 1]

    # Seuillage automatique Otsu sur le canal A
    _, msk = cv2.threshold(
        a_channel,
        0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nettoyage du mask — erase "petits bruits"
    kernel = np.ones((5, 5), np.uint8)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
    msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)

    return msk



def roi_objects(img, msk):
	result = img.copy()

	# Trouver les contours dans le mask
	contours, _ = cv2.findContours(
		msk,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)

	if not contours:
		return result

	# Garder uniquement le plus grand contour = la feuille
	largest = max(contours, key=cv2.contourArea)

	# Dessiner le contour vert
	cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)

	# Dessiner le bounding box bleu
	x, y, w, h = cv2.boundingRect(largest)
	cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

	return result



def analyze_object(img, msk):
	result = img.copy()

	contours, _ = cv2.findContours(
		msk,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)
	if not contours:
		return result

	largest = max(contours, key=cv2.contourArea)

	# Mesures morphologiques
	area        = cv2.contourArea(largest)
	perimeter   = cv2.arcLength(largest, closed=True)
	x, y, w, h  = cv2.boundingRect(largest)
	solidity    = area / (w * h) if (w * h) > 0 else 0

	# Dessine contour Magenta
	cv2.drawContours(result, [largest], -1, (255, 0, 255), 2)

	# Axe principal via ellipse
	if len(largest) >= 5:
		ellipse = cv2.fitEllipse(largest)
		cv2.ellipse(result, ellipse, (255, 0, 255), 2)

	# Affiche mesures sur image
	metrics = [
		f"Area      : {int(area)} px",
		f"Perimeter : {int(perimeter)} px",
		f"Width     : {w} px",
		f"Height    : {h} px",
		f"Solidity  : {solidity:.2f}",
	]
	for i, text in enumerate(metrics):
		cv2.putText(
			result,
			text,
			(10, 20 + i * 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (255, 0, 255), 1
	)

	return result



def pseudolandmarks(img, msk):

	landmark_tolerance = 20

	result = img.copy()
	contours, _ = cv2.findContours(
		msk,
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)
	if not contours:
		return result

	largest = max(contours, key=cv2.contourArea)
	points  = largest[:, 0, :]  # reshape (N, 1, 2) -> (N, 2)

	# Axe X : diviser la feuille en 3 tiers verticaux
	x_min = points[:, 0].min()
	x_max = points[:, 0].max()
	x_third = (x_max - x_min) // 3

	for i in range(4):  # 4 lignes = 3 tiers
		x = x_min + i * x_third
		pts_near = points[np.abs(points[:, 0] - x) < landmark_tolerance]
		if len(pts_near) == 0:
			continue
		top    = pts_near[pts_near[:, 1].argmin()]
		bottom = pts_near[pts_near[:, 1].argmax()]
		mid    = pts_near[np.abs(pts_near[:, 1] - pts_near[:, 1].mean()).argmin()]

		cv2.circle(result, tuple(top),    5, (0, 0, 255),   -1)  # rouge
		cv2.circle(result, tuple(bottom), 5, (0, 165, 255), -1)  # orange
		cv2.circle(result, tuple(mid),    5, (0, 255, 0),   -1)  # vert

	# Axe Y : diviser la feuille en 3 tiers horizontaux
	y_min = points[:, 1].min()
	y_max = points[:, 1].max()
	y_third = (y_max - y_min) // 3

	for i in range(4):
		y = y_min + i * y_third
		pts_near = points[np.abs(points[:, 1] - y) < landmark_tolerance]
		if len(pts_near) == 0:
			continue
		left  = pts_near[pts_near[:, 0].argmin()]
		right = pts_near[pts_near[:, 0].argmax()]
		mid   = pts_near[np.abs(pts_near[:, 0] - pts_near[:, 0].mean()).argmin()]

		cv2.circle(result, tuple(left),  5, (255, 0, 0),   -1)  # bleu
		cv2.circle(result, tuple(right), 5, (255, 0, 255), -1)  # magenta
		cv2.circle(result, tuple(mid),   5, (0, 255, 255), -1)  # cyan

	return result



def color_histogram(img, msk):
	fig, ax = plt.subplots(figsize=(10, 5))

	# Def canaux a analiser
	channels = {
		# BGR
		"blue":( img[:, :, 0], "blue"  ),
		"green": ( img[:, :, 1], "green" ),
		"red": ( img[:, :, 2], "red" ),
		# HSV
		"hue": ( cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0], "purple" ),
		"saturation": ( cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1], "orange" ),
		"value": ( cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2], "yellow" ),
		# LAB
		"lightness": ( cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0], "gray" ),
		"green-magenta": ( cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1], "magenta" ),
		"blue-yellow": ( cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2], "cyan" ),
	}

	for name, (channel, color) in channels.items():
		# Calculer histogramme uniquement sur la zone masquee
		hist = cv2.calcHist(
			[channel],
			[0],
			msk,
			[256],
			[0, 256]
		)
		# Normalisation en %
		hist = hist / hist.sum() * 100
		ax.plot(hist, color=color, label=name, linewidth=1)

	ax.set_xlabel("Pixel intensity")
	ax.set_ylabel("Proportion of pixels (%)")
	ax.legend(loc="upper right", fontsize=8)
	ax.set_title("Color Histogram")

	plt.tight_layout()
	plt.show()



def display_transformations(results):
	n = len(results)
	fig, axes = plt.subplots(2, 3, figsize=(14, 8))
	axes = axes.flatten()

	for i, (title, image) in enumerate(results.items()):
		if len(image.shape) == 2:
			# Image en niveaux de gris (mask)
			axes[i].imshow(image, cmap='gray')
		else:
			# Image BGR -> RGB pour matplotlib !
			axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
			axes[i].set_title(title)
			axes[i].axis('on')

	plt.tight_layout()
	plt.show()



def process_image(img_path):
	img = cv2.imread(img_path)
	if img is None:
		print(f"[ Error ]: cannot read '{img_path}' !")
		sys.exit(1)

	print(f"\n🔬 - - - [ Processing ]: '{os.path.basename(img_path)}'...\n")
	blurred  = gaussian_blur(img)
	msk      = mask(blurred)
	roi      = roi_objects(img, msk)
	analyzed = analyze_object(img, msk)
	landmarks = pseudolandmarks(img, msk)

	results = {
		"Original":        img,
		"Gaussian Blur":   blurred,
		"Mask":            msk,
		"ROI Objects":     roi,
		"Analyze Object":  analyzed,
		"Pseudolandmarks": landmarks,
	}

	display_transformations(results)
	color_histogram(img, msk)
#





def main():
	if len(sys.argv) != 2:
		print("usage: python3 Transformation.py <image_path>")
		return

	path = sys.argv[1]

	if not os.path.isfile(path):
		print(f"[ Error ]: '{path}' is not a valid file Bro !")
		return

	process_image(path)

	return



if __name__ == "__main__":
	main()
