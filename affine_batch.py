import cv2
import numpy as np
import os
import glob

def affine_transform(image_folder, output_folder):

    subfolders = [f.path for f in os.scandir(image_folder) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)

        output_subfolder = os.path.join(output_folder,subfolder_name)
        os.makedirs(output_subfolder,exist_ok=True)

        image_files = glob.glob(os.path.join(subfolder,"*.jpg"))

        for image_file in image_files:
            image = cv2.imread(image_file)

            angles = list(range(0,360,15))

            transformed_images = []

            for angle in angles:
                rotated_image = image.copy()

                matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2),angle,1.0)
                rotated_image = cv2.warpAffine(rotated_image, matrix,(image.shape[1],image.shape[0]),borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

                transformed_images.append(rotated_image)

            shearing_factors = [(0.1,0),(-0.1,0),(0,0.1),(0,-0.1)]

            sheared_images =[]
            
            for tranformed_image in transformed_images:
                for shearing_factor in shearing_factors:
                    shearing_matrix = np.array([[1, shearing_factor[0],0],
                                                [shearing_factor[1],1,0]])
                                                
                    transformed_with_shearing_image =cv2.warpAffine(tranformed_image,shearing_matrix, (image.shape[1],image.shape[0]),borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

                    mirror_image = cv2.flip(transformed_with_shearing_image, 1)
                    sheared_images.extend([transformed_with_shearing_image, mirror_image])

            filename = os.path.splitext(os.path.basename(image_file))[0]
            for i, transformed_image in enumerate(transformed_images):
                output_path = os.path.join(output_subfolder, f"transformed_{filename}_{i}.jpg")
                cv2.imwrite(output_path, transformed_image)
                print(f"Transformed image saved at: {output_path}")

            for i, sheared_image in enumerate(sheared_images):
                output_path = os.path.join(output_subfolder, f"sheared_{filename}_{i}.jpg")
                cv2.imwrite(output_path, sheared_image)
                print(f"Sheared image saved at: {output_path}")


    print("Transformation completed successfully.")

image_folder = "D:/neutral_colored"
output_folder = "D:/siamese training dataset"

os.makedirs(output_folder, exist_ok=True)
affine_transform(image_folder, output_folder)