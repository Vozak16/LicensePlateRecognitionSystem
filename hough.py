import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math


class Hough:
    def __init__(self, image):
        self._image = image
        self._edge_height, self._edge_width = self._image.shape[:2]
        self._edge_height_half, self._edge_width_half = self._edge_height / 2, self._edge_width / 2

    def line_detection_vectorized(self, num_rhos=500, num_thetas=180):
        t_count = num_rhos * 0.25

        d = np.sqrt(np.square(self._edge_height) + np.square(self._edge_width))
        dtheta = 180 / num_thetas
        drho = (2 * d) / num_rhos

        thetas = np.arange(0, 180, step=dtheta)
        rhos = np.arange(-d, d, step=drho)

        cos_thetas = np.cos(np.deg2rad(thetas))
        sin_thetas = np.sin(np.deg2rad(thetas))

        edge_points = np.argwhere(self._image != 0)
        edge_points = edge_points - np.array([[self._edge_height_half, self._edge_width_half]])

        rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))

        accumulator, theta_vals, rho_vals = np.histogram2d(
            np.tile(thetas, rho_values.shape[0]),
            rho_values.ravel(),
            bins=[thetas, rhos]
        )
        accumulator = np.transpose(accumulator)
        lines = list(np.argwhere(accumulator > t_count))

        for _ in range(5):
            lines = self.averaging(lines)

        contours_lines = self.perpendicular_contours(lines, thetas, rhos)
        return self.find_points(contours_lines)

    def averaging(self, lines):
        lines = [[x, True] for x in lines]
        result_lines = []
        for i in range(len(lines)):
            if lines[i][1]:
                averaged = lines[i][0]
                num_coincide = 1
                for j in range(len(lines) - i - 1):
                    delta = abs(lines[i][0][0] - lines[i + j + 1][0][0]) + abs(lines[i][0][1] - lines[i + j + 1][0][1])
                    if delta < 25:
                        averaged[0] += lines[i + j + 1][0][0]
                        averaged[1] += lines[i + j + 1][0][1]
                        lines[i + j + 1][1] = False
                        num_coincide += 1
                result_lines.append([int(x / num_coincide) for x in averaged])
        return result_lines

    def perpendicular_contours(self, lines, thetas, rhos):
        lines = [tuple(i) for i in lines]
        dct_perp = dict()
        for i in lines:
            if i not in dct_perp:
                dct_perp[i] = []
            for j in lines:
                if j not in dct_perp:
                    dct_perp[j] = []
                delta = abs(i[1] - j[1] - 90)
                if delta < 5:
                    dct_perp[i].append(j)
                    dct_perp[j].append(i)

        contours_lines = []
        for first in dct_perp:
            for second in dct_perp[first]:
                for third in dct_perp[second]:
                    for fourth in dct_perp[third]:
                        if fourth in dct_perp[first] and \
                                first != third and \
                                second != fourth and \
                                [third, second, first, fourth] not in contours_lines and \
                                [first, fourth, third, second] not in contours_lines:
                            contours_lines.append([first, second, third, fourth])

        # figure = plt.figure(figsize=(9, 9))
        # subplot4 = figure.add_subplot()
        # subplot4.imshow(self._image)

        new_contours = []
        for contour in contours_lines:
            new_contour = []
            for line in contour:
                y, x = line
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + self._edge_width_half
                y0 = (b * rho) + self._edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                new_contour.append([(y1 - y2) / (x1 - x2 + 0.05), (y2 * x1 - y1 * x2) / (x1 - x2 + 0.05)])
                # subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))
            new_contours.append(new_contour)
        # subplot4.title.set_text("Detected Lines")
        # plt.show()
        return new_contours

    def find_points(self, contours_lines):
        points = []
        for contour in contours_lines:
            lst = []
            for i in range(4):
                k1 = contour[i][0]
                b1 = contour[i][1]
                k2 = contour[(i + 1) % 4][0]
                b2 = contour[(i + 1) % 4][1]
                x = int((b2 - b1) / (k1 - k2))
                y = int((k1 * b2 - b1 * k2) / (k1 - k2))
                if (0 <= x <= self._edge_height) and (0 <= y <= self._edge_height):
                    lst.append(np.array([[x, y]], dtype=np.int32))
            if len(lst) == 4:
                points.append(np.asarray(lst))
        return points
