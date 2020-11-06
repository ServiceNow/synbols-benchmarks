import pandas as pd
import numpy as np


def to_latex(result_str, factor=1., skip_rows=(), skip_cols=(), bold_rows=(), bold_cols=(), float_format="%.1f",
             bold_mask=None, red_mask=None):
    table = []
    for i, line in enumerate(result_str.splitlines()):
        if i in skip_rows:
            continue

        row = []
        for j, cell in enumerate(line.split('\t')):
            if j in skip_cols:
                continue

            try:
                if '±' in cell:
                    mean, std = cell.split('±')
                    format = "%s \std{%s}" % (float_format, float_format)
                    cell = format % (float(mean) * factor, float(std) * factor)
                else:
                    cell = float_format % (float(cell) * factor)

            except ValueError:
                pass

            if (i in bold_rows) or (j in bold_cols) or (bold_mask is not None and bold_mask[i, j]):
                if cell.strip() is not "":
                    cell = "\\bf %s" % cell

            if red_mask is not None and red_mask[i,j]:
                if cell.strip() is not "":
                    cell = "\\rd %s" % cell

            row.append(cell)

        table.append(row)

    n_col = np.max([len(row) for row in table])
    col_format = 'r' + 'c' * (n_col - 1)
    return pd.DataFrame(table[1:], columns=table[0]).to_latex(index=False, escape=False, column_format=col_format)


def supervised_learning():
    result_str = """\
Dataset		MNIST	SVHN	Synbols Default		Camouflage	Natural	Korean	Less Variations
Label Set		10 Digits	10 Digits	48 Symbols	48 Symbols	48 Symbols	48 Symbols	1000 Symbols	1120 Fonts
Dataset Size	N params	60k	100k	100k	1M	100k	100k	100k	100k
version		-	-	October 18	October 18	October 18	October 18	October 18	October 18
		accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy
MLP	72K	98.51 ±0.02	85.04 ±0.21	14.83 ±0.40	63.05 ±0.80	4.08 ±0.08	5.02 ±0.08	0.12 ±0.02	0.11 ±0.03
Conv-4 Flat	138K	99.32 ±0.06	90.74 ±0.27	68.51 ±0.66	89.45 ±0.19	32.35 ±1.51	19.43 ±1.01	1.62 ±0.13	0.21 ±0.04
Conv-4 GAP	112K	99.06 ±0.07	88.32 ±0.21	70.14 ±0.41	90.87 ±0.11	29.60 ±0.55	25.60 ±1.05	33.58 ±4.65	3.16 ±0.38
ResNet-12	7.9M	99.70 ±0.05	96.38 ±0.03	95.43 ±0.12	98.85 ±0.02	90.14 ±0.05	81.21 ±0.46	97.08 ±0.13	39.41 ±0.30
ResNet-12+		99.73 ±0.05	97.19 ±0.04	97.16 ±0.05	99.44 ±0.00	94.09 ±0.07	85.80 ±0.15	98.54 ±0.07	57.42 ±0.50
WRN-28-4		99.64 ±0.06	96.07 ±0.07	93.57 ±0.29	98.88 ±0.04	86.34 ±0.16	73.26 ±0.53	95.79 ±0.51	23.10 ±0.90
WRN-28-4+		99.74 ±0.03	97.30 ±0.05	97.41 ±0.04	99.57 ±0.01	95.55 ±0.25	88.30 ±0.23	99.14 ±0.09	68.42 ±1.11"""

    mean, std = extract_tables(result_str)
    bold_mask = bold_best(mean, std)
    print(to_latex(result_str, skip_rows=[3, 4], skip_cols=[1], bold_rows=[0, 1, 2], bold_cols=[0],
                   float_format="%.2f", bold_mask=bold_mask))


def bold_best(mean, std):
    n_rows = mean.shape[0]
    bold_mask = []
    for col_idx in range(mean.shape[1]):
        if np.isnan(mean[:, col_idx]).all():
            print("bold_best skip col:", col_idx)
            bold_mask.append([False] * n_rows)
            continue

        best_idx = np.nanargmax(mean[:, col_idx])
        threshold = mean[best_idx, col_idx] - 2 * std[best_idx, col_idx]
        bold_mask.append([val > threshold for val in mean[:, col_idx]])

    return np.array(bold_mask).T


def ood():
    result_str = """Dataset		less variation	Synbols Default							default 1k	default 1M	Less Variations
version		Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18	Oct 18
Partition		Stratified	\iid	Stratified	Compositional	Stratified	Stratified	Compositional	Stratified	Stratified	Compositional	Stratified
		Char		Font	Char-Font	Scale	Rotation	Rot-Scale	x-Translation	y-Translation	x-y-Translation	Char
class		888 font	char	char	char	char	char	char	char	char	char	888 font
size	N params	1M	100k	100k	100k	100k	100k	100k	100k	100k	100k	100k
MLP			14.83 ±0.40	14.94 ±0.37		15.59 ±0.35	11.41 ±0.05	7.91 ±0.21	7.96 ±0.06			0.08 ±0.01
Conv-4 Flat			68.51 ±0.66	67.17 ±0.68		71.75 ±0.17	50.54 ±0.31	53.87 ±0.89	44.78 ±0.72			0.24 ±0.01
Conv-4 GAP			70.14 ±0.41	69.67 ±0.18		62.85 ±0.37	48.25 ±0.38	54.88 ±0.57	65.46 ±0.37			2.97 ±0.32
Resnet-12			95.43 ±0.12	94.62 ±0.09		94.10 ±0.13	82.37 ±0.03	90.56 ±0.20	94.62 ±0.19			25.59 ±0.22
Resnet-12+			97.16 ±0.05	96.10 ±0.06		96.34 ±0.07	91.96 ±0.30	94.43 ±0.26	96.84 ±0.07			33.95 ±0.61
WRN-28-4			93.57 ±0.29	93.27 ±0.11		92.18 ±0.16	79.53 ±0.09	87.68 ±0.46	92.99 ±0.07			16.85 ±0.23
WRN-28-4+			97.41 ±0.04	96.41 ±0.08		96.80 ±0.14	91.81 ±0.48	95.16 ±0.06	97.33 ±0.16			35.41 ±2.39"""

    mean, std = extract_tables(result_str)
    bold_mask, red_mask = bold_drop(mean, ref_col_idx=3)
    latex_str = to_latex(result_str, skip_rows=[1, 4, 5], skip_cols=[1, 2, 5, 10, 11], bold_rows=[0, 1, 2, 3, 4, 5],
                   bold_cols=[0], bold_mask=bold_mask, red_mask=red_mask, float_format="%.2f")

    print(latex_str.replace("0.", "."))

def bold_drop(mean, ref_col_idx, small_drop=0.5, big_drop=5):
    n_rows = mean.shape[0]
    bold_mask = []
    red_mask = []
    for col_idx in range(mean.shape[1]):
        if np.isnan(mean[:, col_idx]).all() or col_idx == ref_col_idx:
            print("skip ", col_idx)
            bold_mask.append([False] * n_rows)
            red_mask.append([False] * n_rows)

        else:
            drop = mean[:, ref_col_idx] - mean[:, col_idx]

            bold_mask.append(drop < small_drop)
            red_mask.append(drop > big_drop)

    return np.array(bold_mask).T, np.array(red_mask).T


def al():
    result_str = """	no-noise	label noise	pixel noise	10% missing	out of the box	20% occluded
BALD	0.53465±0.03527	1.64095±0.02552	0.36777±0.01379	0.90761±0.00891	1.50683±0.03069	1.00760±0.02126
P-Entropy	0.49346±0.02083	1.65369±0.02349	0.35039±0.00807	0.93476±0.01599	1.66729±0.02160	0.95656±0.02732
Random	0.66304±0.01387	1.76075±0.00823	0.50866±0.01547	1.01354±0.05167	1.58185±0.03971	 1.18426±0.02218
BALD calibrated	0.61061±0.12013	1.66233±0.02288	0.35408±0.02094	0.89871±0.01080	1.47419±0.02796	0.98633±0.02682
Entropy calibrated	0.48080±0.04409	1.75532±0.01588	0.34877±0.00669	0.99350±0.00354	1.69163±0.02427	0.98408±0.02693"""
    print(to_latex(result_str, bold_rows=[0], bold_cols=[0], float_format="%.2f"))


def few_shot():
    result_str = """\
Meta-Test	Characters		 Fonts	
Meta-Train	Characters	Fonts	Fonts	Characters
ProtoNet	95.68 ±0.42	75.73 ±0.81	72.45 ±1.52	43.13 ±0.42
RelationNet	87.82 ±2.22	57.00 ±1.76	63.75 ±4.82	38.81 ±1.20
MAML	91.07 ±0.69	66.40 ±4.73	68.65 ±1.87	40.68 ±0.24"""
    print(to_latex(result_str, bold_rows=[0, 1], bold_cols=[0], float_format="%.2f"))


def unsupervised():
    result_str = """\
	Character Accuracy			Font Accuracy		
	Solid Pattern	Shades	Camouflage	Solid Pattern	Shades	Camouflage
Deep InfoMax	83.871 ± 0.801	6.524 ± 0.599	4.846 ± 2.009	16.444 ± 0.674	0.307 ± 0.037	0.283 ± 0.097
VAE	63.484 ± 0.968	22.430 ± 2.648	3.850 ± 0.330	2.682 ± 0.151	0.358 ± 0.072	0.176 ± 0.011
HVAE (2 level)	66.724 ± 9.356	28.858 ± 1.172	3.908 ± 0.187	2.713 ± 0.534	0.389 ± 0.097	0.175 ± 0.006
VAE ResNet	74.156 ± 0.373	37.403 ± 0.462	3.329 ± 0.024	4.968 ± 0.077	0.547 ± 0.035	0.166 ± 0.027
HVAE (2 level) ResNet	72.188 ± 0.112	58.361 ± 3.452	3.516 ± 0.162	3.329 ± 0.057	0.727 ± 0.157	0.162 ± 0.016"""
    print(to_latex(result_str, bold_rows=[0, 1], bold_cols=[0], float_format="%.2f"))


def extract_tables(result_str):
    table = []
    for i, line in enumerate(result_str.splitlines()):

        row = []
        for j, cell in enumerate(line.split('\t')):

            try:
                if '±' in cell:
                    mean, std = [float(s) for s in cell.split('±')]

                else:
                    mean = float(cell)
                    std = 0
            except ValueError:
                mean, std = np.nan, np.nan

            row.append((mean, std))

        table.append(row)
    numeric_values = np.array(table)
    return numeric_values[:, :, 0], numeric_values[:, :, 1]


def ood_merge():
    result_str = """\
0.11 ±0.01	1.73 ±0.18	31.66 ±0.71	0.14 ±0.03	15.69 ±0.20	23.33 ±0.54	25.00 ±0.37	20.83 ±0.31	24.74 ±0.45	19.82 ±0.14
0.45 ±0.08	8.61 ±0.10	78.53 ±0.40	0.50 ±0.22	47.55 ±1.38	62.30 ±0.26	77.98 ±0.36	58.52 ±0.82	64.16 ±0.39	47.39 ±0.85
5.22 ±0.35	11.26 ±0.16	77.39 ±0.34	3.47 ±0.10	71.71 ±0.06	57.44 ±0.14	63.85 ±0.81	73.84 ±0.26	74.00 ±0.30	76.38 ±0.23
4.45 ±0.19	7.93 ±0.13	89.71 ±0.17	3.72 ±0.23	89.56 ±0.09	76.60 ±0.30	87.77 ±0.24	82.96 ±0.15	83.88 ±0.30	83.06 ±0.08
15.49 ±0.70	16.77 ±nan	95.50 ±0.08	11.26 ±0.21	92.96 ±0.31	82.65 ±0.08	93.73 ±0.36	95.77 ±0.22	95.93 ±0.13	96.05 ±0.07
24.18 ±0.61	18.40 ±nan	93.55 ±0.16	15.16 ±0.29	89.62 ±0.12	79.24 ±0.58	87.96 ±0.17	95.08 ±0.02	95.04 ±0.07	92.05 ±0.12
2.18 ±0.28	0.11 ±0.00	93.97 ±0.35	2.18 ±0.28	90.37 ±0.42	80.38 ±0.15	90.87 ±0.42	92.90 ±0.26	92.45 ±0.27	93.45 ±0.33"""

    ref_str = """\
0.16 ±0.01	6.44 ±0.07	72.63 ±0.14	77.82 ±4.58	77.96 ±0.05	69.64 ±0.15	79.65 ±4.46	68.99 ±0.08	69.00 ±0.22	77.27 ±4.80
0.53 ±0.11	30.61 ±0.70	91.13 ±0.19	93.74 ±2.76	91.46 ±0.01	88.30 ±0.10	94.37 ±2.66	89.43 ±0.31	89.33 ±0.04	93.52 ±2.84
6.54 ±0.68	58.33 ±2.67	91.85 ±0.17	94.21 ±2.43	89.93 ±0.09	88.56 ±0.08	94.15 ±2.73	90.89 ±0.08	90.87 ±0.06	93.99 ±2.36
8.86 ±0.95	85.04 ±0.60	97.34 ±0.04	97.97 ±0.90	97.50 ±0.13	95.94 ±0.14	98.18 ±0.86	96.29 ±0.04	96.33 ±0.01	97.77 ±0.85
44.86 ±7.21	95.88 ±0.15	99.25 ±0.04	99.29 ±0.26	98.91 ±0.03	98.46 ±0.03	99.32 ±0.32	98.69 ±0.05	98.70 ±0.02	99.18 ±0.26
76.82 ±1.32	96.95 ±0.03	97.98 ±0.52	98.03 ±0.73	97.87 ±0.41	97.11 ±0.59	98.03 ±0.76	97.38 ±0.53	97.45 ±0.56	97.88 ±0.58
7.14 ±5.21	80.19 ±2.09	98.73 ±0.01	98.87 ±0.48	98.29 ±0.06	97.66 ±0.10	99.00 ±0.46	98.06 ±0.05	98.05 ±0.03	98.76 ±0.49"""

    result = extract_tables(result_str)

    ref = extract_tables(ref_str)

    diff = result[:, :, 0] - ref[:, :, 0]
    std = np.sqrt(result[:, :, 1] ** 2 + ref[:, :, 1] ** 2)

    for i in range(diff.shape[0]):
        row = "\t".join(["%.2f ± %.2f" % (diff[i, j], std[i, j]) for j in range(diff.shape[1])])
        print(row)

    print()
    mean_diff = np.mean(diff, axis=1)
    print("\n".join(["%.2f" % val for val in mean_diff]))

    ref_mean = np.mean(ref[:, 2:, 0], axis=1)
    ref_std = np.std(ref[:, 2:, 0], axis=1)

    print("\n".join(["%.2f ± %.2f" % (m, s) for m, s in zip(ref_mean, ref_std)]))


# supervised_learning()
# ood()
al()
# few_shot()
# unsupervised()

# ood_merge()
