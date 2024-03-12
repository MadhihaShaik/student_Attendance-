import os
import cv2
import face_recognition

def load_images_from_folder(folder):
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    images = []
    for file in image_files:
        img_path = os.path.join(folder, file)
        img = face_recognition.load_image_file(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append((file.split('.')[0], img))  # Assuming file name is the person's name
    return images

def recognize_and_draw_faces(image):
    face_loc = face_recognition.face_locations(image)
    if not face_loc:
        print(f"No face detected in the image.")
        return None, image

    encode = face_recognition.face_encodings(image)[0]
    cv2.rectangle(image, (face_loc[0][3], face_loc[0][0]), (face_loc[0][1], face_loc[0][2]), (155, 0, 255), 2)
    return encode, image

def compare_faces(image1, image2):
    try:
        face_loc_img1 = face_recognition.face_locations(image1)
        if not face_loc_img1:
            print(f"No face detected in image1.")
            return None, None, image1, image2

        encode_img1 = face_recognition.face_encodings(image1)[0]
        cv2.rectangle(image1, (face_loc_img1[0][3], face_loc_img1[0][0]), (face_loc_img1[0][1], face_loc_img1[0][2]), (155, 0, 255), 2)

        face_loc_img2 = face_recognition.face_locations(image2)
        if not face_loc_img2:
            print(f"No face detected in image2.")
            return None, None, image1, image2

        encode_img2 = face_recognition.face_encodings(image2)[0]
        cv2.rectangle(image2, (face_loc_img2[0][3], face_loc_img2[0][0]), (face_loc_img2[0][1], face_loc_img2[0][2]), (155, 0, 255), 2)

        results = face_recognition.compare_faces([encode_img1], encode_img2)
        face_dis = face_recognition.face_distance([encode_img1], encode_img2)

        return results, face_dis, image1, image2

    except IndexError:
        print("Error: Face not detected in one of the images.")
        return None, None, image1, image2

def main():
    folder_path = 'Images_Attendance'
    images = load_images_from_folder(folder_path)

    for i in range(len(images) - 1):
        for j in range(i + 1, len(images)):
            name1, original_img1 = images[i]
            name2, original_img2 = images[j]

            encode_img1, img1 = recognize_and_draw_faces(original_img1)
            encode_img2, img2 = recognize_and_draw_faces(original_img2)

            results, face_dis, _, _ = compare_faces(original_img1, original_img2)

            if results is not None and face_dis is not None:
                print(f"Comparison between {name1} and {name2}: {results} - Distance: {round(face_dis[0], 2)}")

                cv2.putText(img2, f'{results} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                cv2.imshow(f'{name1} vs {name2}', img2)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
