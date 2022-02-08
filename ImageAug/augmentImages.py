import imgaug.augmenters as iaa
import cv2
import os
import glob
import uuid


def renameImages(folderPath, img_label):
    os.chdir(folderPath)
    directory = os.listdir(path)
    for num, file_name in enumerate(directory):
        new_name = img_label + '_' + str(num + 1) + ".jpg"
        src = path + '\\' + file_name
        dst = path + '\\' + new_name
        os.rename(src, dst)
        # if num == len(directory):
        #     return True
    print('\nAll Images In Dataset renamed...')

def imageAugmentation(folderPath, img_label):
    # Initialise variables and paths
    images = []
    augmented = []
    filenames = []

    images_path = glob.glob(path + "\\*.jpg")

    # 1. Load Dataset
    for image in images_path:
        img = cv2.imread(image)
        images.append(img)
    print('Dataset Loaded...')

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
    print('Dataset Augmented...')
    # # 3. Show Images
    # for img in augmented_images:
    #     augmented.append(img)
    #     # cv2.imshow("Image", img)
    #     # cv2.waitKey(0)
    #

    for i in range(0, len(augmented_images)):
        aug_name = img_label + '_' + str(i + 1) + "_" + str(uuid.uuid1()) + ".jpg"
        filenames.append(aug_name)
    print('Augmented Files Named...\n')
    # for file in filenames:
    #     print(file)

    # 4. Save Augmented Images
    os.chdir(path)
    print(f'DIR: {path}')
    print(f'Images in DIR Before: {len(os.listdir(path))}')

    for count, file in enumerate(filenames):
        cv2.imwrite(file, augmented_images[count])

    print(f'Images in DIR After: {len(os.listdir(path))}')


########### Main ############

# Change to filepath containing images
path = r'D:\Jeremy\Desktop\Tensorflow\Trauma_ObjectDetection\Images\LacerationWound'

# Change to specific label of class
label = 'LacerationWound'

# Augmenting and Renaming of Dataset
imageAugmentation(path, label)
renameImages(path, label)
