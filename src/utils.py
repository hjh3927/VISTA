import numpy as np
from scipy.optimize import least_squares

def bezier_curve(t, P0, P1, P2, P3):
    """
    计算三阶贝塞尔曲线上的点，t 为 [0,1] 间的参数数组
    """
    t = np.array(t).reshape(-1, 1)
    return (1 - t)**3 * P0 + 3*(1 - t)**2 * t * P1 + 3*(1 - t) * t**2 * P2 + t**3 * P3

def fit_bezier_segment(points):
    """
    拟合单段贝塞尔曲线，返回四个控制点
    """
    n = len(points)
    t = np.linspace(0, 1, n)
    P0 = points[0]
    P3 = points[-1]
    P1_guess = points[int(n/3)]
    P2_guess = points[int(2*n/3)]
    def residuals(params):
        P1 = params[:2]
        P2 = params[2:]
        curve_points = bezier_curve(t, P0, P1, P2, P3)
        return (curve_points - points).flatten()
    result = least_squares(residuals, np.concatenate([P1_guess, P2_guess]))
    P1 = result.x[:2]
    P2 = result.x[2:]
    return P0, P1, P2, P3

def compute_error(points, P0, P1, P2, P3):
    """
    计算贝塞尔曲线与点集的最大误差
    """
    t = np.linspace(0, 1, len(points))
    curve_points = bezier_curve(t, P0, P1, P2, P3)
    errors = np.linalg.norm(curve_points - points, axis=1)
    return np.max(errors)

def point_to_line_distance(points, p1, p2):
    """
    计算点集到直线的距离
    """
    p1, p2, points = map(np.array, [p1, p2, points])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.zeros(len(points))
    point_vec = points - p1
    t = np.clip(np.dot(point_vec, line_vec) / (line_len**2), 0, 1)
    projections = p1 + t[:, None] * line_vec
    distances = np.linalg.norm(points - projections, axis=1)
    return distances
