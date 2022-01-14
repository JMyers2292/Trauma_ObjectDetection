import imgaug.augmenters as iaa
import cv2
import os
import glob

# Initialise variables and paths
images = []
augmented = []
filenames = []

# Change to filepath containing images
path = r'C:\Users\Jeremy\Desktop\AugmentedImages'
images_path = glob.glob(path + "\\*.jpg")

# 1. Load Dataset
for image_path in images_path:
    img = cv2.imread(image_path)
    images.append(img)

# 2. Image Augmentation
augmentation = iaa.Sequential([
    # 1. Flip
    iaa.Fliplr(0.5),  # Vertical direction, % of flipping img
    iaa.Flipud(0.5),  # Horizontal direction, % of flipping img

    # 2. Affine
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Moves img from range set in x|y coordinate
               rotate=(-30, 30),  # Rotate img dependant on range set
               scale=(0.5, 1.5)  # Zoom in or out depending on scale set
               ),

    # 3. Multiply
    iaa.Multiply((0.8, 1.2)),  # Makes img brighter or darker

    # 4. Linear Contrast
    iaa.LinearContrast((0.9, 1.4)),  # Make img contrast more or less

    # Perform methods below only sometimes
    iaa.Sometimes(0.5,
                  # 5. Gaussian Blur
                  iaa.GaussianBlur((0.0, 1.0))
                  )
])

augmented_images = augmentation(images=images)

# 3. Show Images
for img in augmented_images:
    augmented.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

for i in range(0, len(augmented)):
    filenames.append("augmented_" + str(i+1) + ".jpg")

# 4. Save Augmented Images
os.chdir(path)
print(f'DIR: {path}')
print(f'Images in DIR Before: {len(os.listdir(path))}')

for count, file in enumerate(filenames):
    cv2.imwrite(file, augmented[count])

print(f'Images in DIR After: {len(os.listdir(path))}')
