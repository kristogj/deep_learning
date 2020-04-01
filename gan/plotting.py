# Generate plots
import pandas as pd
import seaborn as sns


# Usage example:
# def test():
#     data1 = [1.2, 2.3, 3.4, 4.5, 5.6]
#     data2 = [1.1, 2.2, 3.3, 4.4, 5.5]
#
#     data = [data1, data2]
#     generate_lineplot(data, "Y_axis_title", "test")

def generate_lineplot(data, y_title, filename, data_type='Dataset', type_names=('Generator', 'Discriminator')):
    columns = ['Epoch', data_type, y_title]
    df = pd.DataFrame(columns=columns)

    for i in range(len(data)):
        for j in range(len(data[i])):
            df.loc[len(df)] = [j + 1, type_names[i], data[i][j]]

    sns.set(style="darkgrid")
    plt = sns.lineplot(data=df, x='Epoch', y=y_title, style=data_type, hue=data_type)
    plt.figure.savefig(filename)
    plt.figure.clf()
