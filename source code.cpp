import cap as cap
import imutils as imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt


def midPointCircleDraw(x_centre, y_centre, r):
    image = cv2.imread('C:/Users/Saniya/Desktop/img.png')
    img = cv2.resize(image, (800, 800))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img[x_centre - 3: x_centre + 3, y_centre - 3: y_centre + 3, 0] = 0
    img[x_centre - 3: x_centre + 3, y_centre - 3: y_centre + 3, 1] = 0
    img[x_centre - 3: x_centre + 3, y_centre - 3: y_centre + 3, 2] = 0

    # px = img[400,400]
    # print(px)

    # plt.imshow(img, interpolation='none')
    # plt.show()

    img = cv2.putText(img, str(x_centre) + ", " + str(y_centre), (x_centre, y_centre), cv2.FONT_HERSHEY_SIMPLEX,
                      0.8, (0, 0, 0), 2, cv2.LINE_AA)

    img = cv2.putText(img, "Press Q to exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    x = r
    y = 0

    plt.imshow(img)
    plt.pause(0.7)

    img[x + x_centre, y + y_centre, 0] = 0
    img[x + x_centre, y + y_centre, 1] = 0
    img[x + x_centre, y + y_centre, 2] = 0

    # When radius is zero only a single
    # point be printed
    if r > 0:
        # print("(", x + x_centre, ", ",
        #       -y + y_centre, ")",
        #       sep="", end="")
        img[x + x_centre, -y + y_centre, 0] = 0
        img[x + x_centre, -y + y_centre, 1] = 0
        img[x + x_centre, -y + y_centre, 2] = 0
        # print("(", y + x_centre, ", ",
        #       x + y_centre, ")",
        #       sep="", end="")
        img[y + x_centre, x + y_centre, 0] = 0
        img[y + x_centre, x + y_centre, 1] = 0
        img[y + x_centre, x + y_centre, 2] = 0
        # print("(", -y + x_centre, ", ",
        #       x + y_centre, ")", sep="")
        img[-y + x_centre, x + y_centre, 0] = 0
        img[-y + x_centre, x + y_centre, 1] = 0
        img[-y + x_centre, x + y_centre, 2] = 0

    # Initialising the value of P
    P = 1 - r

    while x > y:

        y += 1

        # Mid-point inside or on the perimeter
        if P <= 0:
            P = P + 2 * y + 1

        # Mid-point outside the perimeter
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1

        # All the perimeter points have
        # already been printed
        if (x < y):
            break

        # Printing the generated point its reflection
        # in the other octants after translation
        # print("(", x + x_centre, ", ", y + y_centre,
        #       ")", sep="", end="")
        img[x + x_centre, y + y_centre, 0] = 0
        img[x + x_centre, y + y_centre, 1] = 0
        img[x + x_centre, y + y_centre, 2] = 0
        # print("(", -x + x_centre, ", ", y + y_centre,
        #       ")", sep="", end="")
        img[-x + x_centre, y + y_centre, 0] = 0
        img[-x + x_centre, y + y_centre, 1] = 0
        img[-x + x_centre, y + y_centre, 2] = 0
        # print("(", x + x_centre, ", ", -y + y_centre,
        #       ")", sep="", end="")
        img[x + x_centre, -y + y_centre, 0] = 0
        img[x + x_centre, -y + y_centre, 1] = 0
        img[x + x_centre, -y + y_centre, 2] = 0
        # print("(", -x + x_centre, ", ", -y + y_centre,
        #       ")", sep="")
        img[-x + x_centre, -y + y_centre, 0] = 0
        img[-x + x_centre, -y + y_centre, 1] = 0
        img[-x + x_centre, -y + y_centre, 2] = 0

        # If the generated point on the line x = y then
        # the perimeter points have already been printed
        if x != y:
            # print("(", y + x_centre, ", ", x + y_centre,
            #       ")", sep="", end="")
            img[y + x_centre, x + y_centre, 0] = 0
            img[y + x_centre, x + y_centre, 1] = 0
            img[y + x_centre, x + y_centre, 2] = 0
            # print("(", -y + x_centre, ", ", x + y_centre,
            #       ")", sep="", end="")
            img[-y + x_centre, x + y_centre, 0] = 0
            img[-y + x_centre, x + y_centre, 1] = 0
            img[-y + x_centre, x + y_centre, 2] = 0
            # print("(", y + x_centre, ", ", -x + y_centre,
            #       ")", sep="", end="")
            img[y + x_centre, -x + y_centre, 0] = 0
            img[y + x_centre, -x + y_centre, 1] = 0
            img[y + x_centre, -x + y_centre, 2] = 0
            # print("(", -y + x_centre, ", ", -x + y_centre,
            #       ")", sep="")
            img[-y + x_centre, -x + y_centre, 0] = 0
            img[-y + x_centre, -x + y_centre, 1] = 0
            img[-y + x_centre, -x + y_centre, 2] = 0

    plt.imshow(img, interpolation='none')
    plt.show()
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# if __name__ == '__main__':

webcam = cv2.VideoCapture(0)
count = 0
img_counter = 0

while True:
    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    imageFrame = imutils.resize(imageFrame, width=800, height=800)

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            # cv2.putText(imageFrame, "Red Colour", (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             (0, 0, 255))

            # count = count + 1;

            if x > 700 or y > 700 or x < 100 or y < 100:
                cv2.putText(imageFrame, str(x) + ", " + str(y), (x - 1, y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255))
                cv2.putText(imageFrame, "These coordinates are out of range.",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            else:
                cv2.putText(imageFrame, str(x) + ", " + str(y), (x - 1, y - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255))

            img_name = "opencv_frame_{}.png".format(img_counter)

            cv2.imread(img_name)
            cv2.imshow(img_name, imageFrame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

            if not(x > 700 or y > 700 or x < 100 or y < 100):
                midPointCircleDraw(x, y, 100)

            cv2.imwrite(img_name, imageFrame)
            print("{} written!".format(img_name))
            img_counter += 1

            print(x, "  ", y)

            # cv2.putText(imageFrame, x, (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            #             (0, 0, 255))

            # Creating contour to track green color
    # contours, hierarchy = cv2.findContours(green_mask,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    #
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if (area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         imageFrame = cv2.rectangle(imageFrame, (x, y),
    #                                    (x + w, y + h),
    #                                    (0, 255, 0), 2)
    #
    #         cv2.putText(imageFrame, "Green Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1.0, (0, 255, 0))

    # Creating contour to track blue color
    # contours, hierarchy = cv2.findContours(blue_mask,
    #                                        cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if (area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         imageFrame = cv2.rectangle(imageFrame, (x, y),
    #                                    (x + w, y + h),
    #                                    (255, 0, 0), 2)
    #
    #         cv2.putText(imageFrame, "Blue Colour", (x, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1.0, (255, 0, 0))

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
