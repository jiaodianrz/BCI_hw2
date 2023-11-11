import mat73
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score


def calculate_angle(vector1, vector2):
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    dot_product = np.dot(vector1, vector2)
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

    # 使用叉乘判断向量之间的方向
    cross_product = np.cross(vector1, vector2)
    if cross_product < 0:
        angle_radians = 2 * np.pi - angle_radians

    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def cal_cur_angle(cur_pos):
    cur_x = cur_pos[:, 0]
    cur_y = cur_pos[:, 1]
    vec_x = np.diff(cur_x)
    vec_y = np.diff(cur_y)
    cur_vec = np.concatenate([vec_x[:, np.newaxis], vec_y[:, np.newaxis]], axis=1)
    angles = []
    for i in range(cur_vec.shape[0]-1):
        angles.append(calculate_angle(cur_vec[i], cur_vec[i+1]))

    return np.array(angles)


# 返回每个角度区间内的时间戳
def split_angles(intervals, angles, t_stamps):
    corr_times = []
    for interval in intervals:
        _corr_times = []
        for i, ang in enumerate(angles):
            if interval[0] <= ang < interval[1] or interval[0] <= ang-360 <= interval[1]:
                _corr_times.append(t_stamps[i+1])
        corr_times.append(np.array(_corr_times))

    return corr_times

# 返回时间戳
def find_time_stamp(t_stamps, spike_time):
    left = 0
    right = len(t_stamps) - 1

    # 处理特殊情况
    if spike_time < t_stamps[left] or  spike_time > t_stamps[right]:
        return None

    while left <= right:
        mid = (left + right) // 2

        if t_stamps[mid] == spike_time:
            return t_stamps[mid]
        elif t_stamps[mid] < spike_time:
            left = mid + 1
        else:
            right = mid - 1

    # 返回最接近的数
    if abs(spike_time - t_stamps[left]) < abs(spike_time - t_stamps[right]):
        return t_stamps[left]
    else:
        return t_stamps[right]


def spikes_of_angle(spike_times, corr_times, t_stamps):
    spikes_num = [0]*len(corr_times)
    for spike_time in spike_times:
        spike_stamp = find_time_stamp(t_stamps, spike_time)
        if spike_stamp is None:
            continue
        for i, _corr_time in enumerate(corr_times):
            if len(np.where(_corr_time == spike_stamp)[0]) > 0:
                spikes_num[i] += 1
    corr_times_num = np.array([len(_corr_times) for _corr_times in corr_times])
    return np.array(spikes_num), corr_times_num


def generate_intervals(interval_num, width):
    interval_len = 360 // interval_num
    intervals = [[i*interval_len-width, i*interval_len+width] for i in range(0, interval_num)]
    x_axis = np.arange(0, 360, interval_len)

    return x_axis, intervals


def cal_spikes_num_ratio(interval_num, spk_l, spk_r, spks, corr_times, t_stamps):
    spikes_num = np.zeros(interval_num)
    corr_times_num = np.zeros(interval_num)
    for ch_spks in spks:
        for neu_spks in ch_spks[1:]:
            if neu_spks is not None and spk_l <= neu_spks.size <= spk_r:
                _spikes_num, _corr_times_num = spikes_of_angle(neu_spks, corr_times, t_stamps)
                spikes_num = spikes_num + _spikes_num
                corr_times_num = corr_times_num + _corr_times_num
    spikes_num_ratio = 1000 * spikes_num / corr_times_num

    return spikes_num_ratio


def analysis_fit_tuning_curve(spk_l, spk_r, spks, corr_times, t_stamps, original_spikes_ratio):
    mse, r2 = [], []
    original_spikes_ratio = original_spikes_ratio / np.sum(original_spikes_ratio)
    for ch_spks in spks:
        for neu_spks in ch_spks[1:]:
            if neu_spks is not None and spk_l <= neu_spks.size <= spk_r:
                spikes_num, corr_times_num = spikes_of_angle(neu_spks, corr_times, t_stamps)
                spikes_ratio = spikes_num / corr_times_num
                if np.sum(spikes_ratio) > 0:
                    spikes_ratio = spikes_ratio / np.sum(spikes_ratio)
                mse.append(mean_squared_error(spikes_ratio, original_spikes_ratio))
                r2.append(r2_score(spikes_ratio, original_spikes_ratio))

    mse = np.array(mse)
    r2 = np.array(r2)

    mse_bins = np.around(np.arange(0, 0.07, 0.01), decimals=3)
    r2_bins = np.around(np.arange(-0.5, 0.5, 0.1), decimals=1)

    plot_hist(mse, mse_bins, 'Fitting MSE', 'MSE', 'Frequency')
    plot_hist(r2, r2_bins, 'Fitting R^2', 'R^2', 'Frequency')


def plot_hist(data, bins, title, x_title, y_title):

    hist, _ = np.histogram(data, bins)
    hist = hist / np.sum(hist)

    plt.bar(range(len(hist)), hist, align='center')
    plt.xticks(range(len(hist)), bins[:-1])

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    # 显示图表
    plt.show()


def plot_tuning_curve(x_axis, spike_times_ratio):
    interp_func = interp1d(x_axis, spike_times_ratio, kind='cubic')

    x_smooth = np.linspace(min(x_axis), max(x_axis), 100)
    y_smooth = interp_func(x_smooth)
    plt.scatter(x_axis, spike_times_ratio, label='Scatter Plot')

    plt.plot(x_smooth, y_smooth, color='red', label='Smooth Curve')

    plt.title('Tuning Curve')
    plt.xlabel('angle')
    plt.ylabel('ratio')

    plt.legend()
    plt.show()