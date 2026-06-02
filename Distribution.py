import os
import sys
import matplotlib.pyplot as plt



def get_img_counts(dir_path: str):
	counts ={}
	valid_img_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

	for item in sorted(os.listdir(dir_path)):

		item_path = os.path.join(dir_path, item)
		if os.path.isdir(item_path):
			images = []
			for file in os.listdir(item_path):
				if file.endswith(valid_img_extensions):
					images.append(file)
			counts[item] = len(images)
	
	return counts



def contains_images(path, valid_extensions):
    for f in os.listdir(path):
        if f.endswith(valid_extensions):
            return True
    return False



def plot_distribution(counts, plant_name):
	if sum(counts.values()) == 0:
		print("[ Error ]: all counts are zero, nothing to Plot !")
		return

	labels = list(counts.keys())
	values = list(counts.values())
	colors = plt.cm.Set2.colors[:len(labels)]

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
	fig.suptitle(f"{plant_name} class distribution", fontsize=14)

	# - - - [ Pie Chart ] - - -
	ax1.pie(
		values,
		labels=labels,
		colors=colors,
		autopct='%1.1f%%',
		startangle=90
	)

	# - - - [ Bar Chart ] - - -
	bars = ax2.bar(labels, values, color=colors)
	ax2.set_ylabel("Number of images")
	ax2.set_xticks(range(len(labels)))
	ax2.set_xticklabels(labels, rotation=45, ha='right')
	for bar, value in zip(bars, values):
		ax2.text(
			bar.get_x() + bar.get_width() / 2,
			bar.get_height() + 10,
			str(value),
			ha='center',
			va='bottom',
			fontsize=9
		)

	# - - - [ Display ] - - -
	plt.tight_layout()
	plt.show()
#





def main():

	if len(sys.argv) != 2:
		print("usage: python3 Distribution.py <directory_path>")
		return


	dir_path = sys.argv[1]
	if not os.path.isdir(dir_path):
		print(f"[ Error ]:'{dir_path}' is not a valid directory !")
		return

	# - - - [ Only Img in Directory ] - - -
	valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
	if contains_images(dir_path, valid_extensions):
		count = len([f for f in os.listdir(dir_path) if f.endswith(valid_extensions)])
		folder_name = os.path.basename(os.path.abspath(dir_path))
		print(f"\n📁 [ '{folder_name}' ] contains {count} images \n\t(no subfolders to display !)")
		return

	# - - - [ One lvl depth Dir ] - - -
	subdirs = [
		item for item in sorted(os.listdir(dir_path))
		if os.path.isdir(os.path.join(dir_path, item))
	]
	first_subdir = os.path.join(dir_path, subdirs[0]) if subdirs else ""
	if first_subdir and contains_images(first_subdir, valid_extensions):
		plant_name = os.path.basename(os.path.abspath(dir_path))
		counts = get_img_counts(dir_path)
		print(f"\n📊 - - - [ {plant_name} - Distribution ] - - - 📊")
		for class_name, count in counts.items():
			print(f"\t{class_name:30s}: {count}\timages")
		print(f"\t{'TOTAL':30s}: {sum(counts.values())} images")
		plot_distribution(counts, plant_name)

	# - - - [ Multi lvl depth Dir ] - - -
	else:
		for plant_dir in subdirs:
			plant_path = os.path.join(dir_path, plant_dir)
			counts = get_img_counts(plant_path)
			if not counts or sum(counts.values()) == 0:
				continue
			print(f"\n📊 - - - [ {plant_dir} - Distribution ] - - - 📊")
			for class_name, count in counts.items():
				print(f"\t{class_name:30s}: {count}\timages")
			print(f"\t{'TOTAL':30s}: {sum(counts.values())} images")
			plot_distribution(counts, plant_dir)


	return



if __name__ == "__main__":
	main()
