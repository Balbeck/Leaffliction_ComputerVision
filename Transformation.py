import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from plantcv import plantcv as pcv



def gaussian_blur(img):
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	pcv.params.debug = None

	blurred_rgb = pcv.gaussian_blur(img=img_rgb, ksize=(5, 5), sigma_x=0)

	return cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)



def mask(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pcv.params.debug = None

    # Convertir en LAB et extraire le canal A (vert vs rouge)
    a_channel = pcv.rgb2gray_lab(rgb_img=img_rgb, channel='a')

    # Seuillage Otsu — "dark" = inverse -> feuille en blanc, fond en noir
    msk = pcv.threshold.otsu(gray_img=a_channel, object_type='dark')

    # Nettoyage — CLOSE puis OPEN
    kernel = pcv.get_kernel(size=(5, 5), shape="rectangle")
    msk = pcv.closing(gray_img=msk, kernel=kernel)
    msk = pcv.opening(gray_img=msk, kernel=kernel)

    return msk



def roi_objects(img, msk):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    pcv.params.debug = None

    # Def ROI qui couvre toute img
    roi = pcv.roi.rectangle(img=img_rgb, x=0, y=0, h=h, w=w)

    # Filtrer le mask par le ROI -> garde le plus grand objet
    filtered_mask = pcv.roi.filter(mask=msk, roi=roi, roi_type='largest')

    # draw result on img
    result = img.copy()
    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (255, 0, 0), 2)

    return result



def analyze_object(img, msk):
	# PlantCV attend du RGB
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Mask en 32-bit pour PlantCV
	labeled_mask = msk.astype(np.int32)

	pcv.outputs.clear()
	pcv.params.sample_label = "plant"
	pcv.params.debug = None  # pas de debug intempestif
	pcv.params.verbose = False

	shape_image = pcv.analyze.size(
		img=img_rgb,
		labeled_mask=labeled_mask,
		n_labels=1
	)

	obs = pcv.outputs.observations.get('plant_1', {})
	metrics = {
		'area':         obs.get('area',         {}).get('value', 'N/A'),
		'perimeter':    obs.get('perimeter',     {}).get('value', 'N/A'),
		'width':        obs.get('width',         {}).get('value', 'N/A'),
		'height':       obs.get('height',        {}).get('value', 'N/A'),
		'solidity':     obs.get('solidity',      {}).get('value', 'N/A'),
		'longest_path': obs.get('longest_path',  {}).get('value', 'N/A'),
	}

	print(f"\n📐 Shape Analysis :")
	for k, v in metrics.items():
		print(f"   {k:15s}: {v}")

	# Return img annotation BGR pour matplotlib/OpenCV
	return cv2.cvtColor(shape_image, cv2.COLOR_RGB2BGR)



def pseudolandmarks(img, msk):
	# PlantCV attend du RGB
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	result = img.copy()

	pcv.params.debug = None
	pcv.params.verbose = False
	pcv.params.sample_label = "plant"

	#Axe X : top, bottom, center_v
	top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
		img=img_rgb,
		mask=msk
	)

	# Axe Y : left, right, center_h
	left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
		img=img_rgb,
		mask=msk
	)

	def draw_points(points, color):
		if points is None:
			return
		for pt in points:
			x, y = int(pt[0][0]), int(pt[0][1])
			cv2.circle(result, (x, y), 5, color, -1)

	# Axe X
	draw_points(top,      (0, 0, 255))    # rouge
	draw_points(bottom,   (0, 165, 255))  # orange
	draw_points(center_v, (0, 255, 0))    # vert

	# Axe Y
	draw_points(left,     (255, 0, 0))    # bleu
	draw_points(right,    (255, 0, 255))  # magenta
	draw_points(center_h, (0, 255, 255))  # cyan

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



def process_directory(src, dst, use_mask):
	valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

	files = [
		f for f in os.listdir(src)
		if f.endswith(valid_extensions)
	]

	if not files:
		print(f"[ Error ]: no images found in '{src}' !")
		sys.exit(1)

	os.makedirs(dst, exist_ok=True)

	total = len(files)
	print(f"\n🔬 Processing '{os.path.basename(src)}' — {total} images\n")

	for i, filename in enumerate(files, 1):
		img_path = os.path.join(src, filename)
		img = cv2.imread(img_path)
		if img is None:
			print(f"  ⚠️  skipping '{filename}' — cannot read")
			continue

		blurred   = gaussian_blur(img)
		msk       = mask(blurred)

		results = {
			"Original":        img,
			"Gaussian_Blur":   blurred,
			"Mask":            msk,
			"ROI_Objects":     roi_objects(img, msk),
			"Analyze_Object":  analyze_object(img, msk),
			"Pseudolandmarks": pseudolandmarks(img, msk),
		}

		# Save Transformations !
		base, ext = os.path.splitext(filename)
		for name, transformed in results.items():
			if use_mask and name == "Original":
				continue
			out_path = os.path.join(dst, f"{base}_{name}{ext}")
			cv2.imwrite(out_path, transformed)

		print(f"  [{i:>4}/{total}] ✅ {filename}")


	print(f"\n✅ Done — results saved in '{dst}'\n")

	return



def parse_args():
	parser = argparse.ArgumentParser(
	prog="Transformation.py",
	description="[ 🔬 Leaf image transformation pipeline ^^ ]",
	formatter_class=argparse.RawTextHelpFormatter,
	epilog="""
		examples:
		python3 Transformation.py image.JPG
		python3 Transformation.py -src ./Apple/apple_healthy/ -dst ./dst/
		python3 Transformation.py -src ./Apple/apple_healthy/ -dst ./dst/ -mask
	"""
	)

	# - - -[ Mode - Simple Img ] - - -
	parser.add_argument(
		"image",
		nargs="?", # optionnel
		help="path to a single image"
	)

	# - - - [ Mode - Full Directory ] - - -
	parser.add_argument(
		"-src",
		metavar="DIR",
		help="source directory with images"
	)
	parser.add_argument(
		"-dst",
		metavar="DIR",
		help="destination directory for results"
	)
	parser.add_argument(
		"-mask",
		action="store_true",  # flag bool
		help="save only masked transformations"
	)

	return parser.parse_args()
#





def main():

	args = parse_args()

	# [ Simple Img ]
	if args.image:
		if not os.path.isfile(args.image):
			print(f"[ Error ]: '{args.image}' is not a valid file !")
			return

		process_image(args.image)


	# [ Full Directory ]
	elif args.src and args.dst:
		if not os.path.isdir(args.src):
			print(f"[ Error ]: '{args.src}' is not a valid directory !")
			return

		process_directory(args.src, args.dst, args.mask)


	else:
		print("[ Error ]: provide an image path or use -src and -dst")
		print("\t- 'python3 Transformation.py -h'  -> for help Bro !")

	return


if __name__ == "__main__":
	main()
