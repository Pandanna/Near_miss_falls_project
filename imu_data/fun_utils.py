import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, Span, BoxAnnotation
from bokeh.plotting import figure, show, output_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score, recall_score, f1_score, precision_score, make_scorer
from sklearn.svm import OneClassSVM


# Abilita Bokeh in Jupyter Notebook
output_notebook()

def correct_json(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
    if content.rstrip()[-1] == ",":
        corrected = content.rstrip(",\n") + "]"
        with open(file_path, 'w') as f:
            f.write(corrected)


def plot_time_data(data, time_column, value_columns, tag_column='Tag', time_unit='second', 
                   xlabel='Time', ylabel='Values', title='Time Series Plot'):
    
    # Conversione nel tempo desiderato
    time_factor = 1
    if time_unit == 'second':
        time_factor = 1000
    elif time_unit == 'minute':
        time_factor = 60000

    # Normalizzazione del tempo
    data['normalized_time'] = (data[time_column] - data[time_column].min()) / time_factor
    
    source = ColumnDataSource(data)

    # Creazione del grafico
    p = figure(height=600, width=1300, title=title, tools="pan,box_zoom,reset,hover,save")
    p.xaxis.axis_label = f"{xlabel} ({time_unit})"
    p.yaxis.axis_label = ylabel

    # Lista semplice di colori
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Disegna le linee
    for i, col in enumerate(value_columns):
        color = colors[i % len(colors)]  # Usa i colori in modo ciclico
        p.line(x='normalized_time', y=col, source=source, line_width=2, color=color, legend_label=col)

    # Evidenzia le aree con Tag = -1
    if tag_column in data.columns:
        segments = []  # Per memorizzare [start, end] dei segmenti
        current_segment = None

        for i in range(len(data)):
            tag = data[tag_column].iloc[i]
            if tag == -1:
                # Inizia un nuovo segmento, se necessario
                if current_segment is None:
                    current_segment = [data['normalized_time'].iloc[i], None]
            else:
                # Termina il segmento corrente
                if current_segment and current_segment[1] is None:
                    current_segment[1] = data['normalized_time'].iloc[i]
                    segments.append(current_segment)
                    current_segment = None

        # Se l'ultimo segmento termina con -1
        if current_segment and current_segment[1] is None:
            current_segment[1] = data['normalized_time'].iloc[-1]
            segments.append(current_segment)

        # Aggiungi le annotazioni per ogni segmento
        for start, end in segments:
            box = BoxAnnotation(left=start, right=end, fill_alpha=0.2, fill_color="magenta")
            p.add_layout(box)

    # Configura la legenda
    p.legend.title = "Legend"
    p.legend.location = "top_left"
    p.legend.click_policy = "hide" 

    # Rimuovi la colonna temporanea
    data.drop(columns=['normalized_time'], inplace=True)

    # Mostra il grafico
    show(p)


def print_acc(imu_data_df, unit):
    plot_time_data(
        data=imu_data_df,
        time_column='Timestamp',
        value_columns=['AccX', 'AccY', 'AccZ'],
        time_unit=unit,
        xlabel='Time',
        ylabel='Acceleration',
        title='Acceleration over Time with Axis Lines'
    )


def print_gyro(imu_data_df, unit):
    plot_time_data(
        data=imu_data_df,
        time_column='Timestamp',
        value_columns=['GyroX', 'GyroY', 'GyroZ'],
        tag_column='Tag', 
        time_unit=unit,
        xlabel='Time',
        ylabel='Gyroscope',
        title='Gyroscope over Time with Axis Lines'
    )


def print_ang(imu_data_df, unit):
    plot_time_data(
        data=imu_data_df,
        time_column='Timestamp',
        value_columns=['AngX', 'AngY', 'AngZ'],
        tag_column='Tag',  
        time_unit=unit,
        xlabel='Time',
        ylabel='Angle',
        title='Angle over Time with Axis Lines'
    )


def print_grafici(imu_data_df, unit='second'):
    print_acc(imu_data_df, unit)
    print_gyro(imu_data_df, unit)
    print_ang(imu_data_df, unit)

    #roll angolazione laterale
    #pitch angolazione verso davanti e verso dietro
    #yaw cambio di posizione del corpo


def get_and_show(file_path, show=False):
    correct_json(file_path)

    df = pd.read_json(file_path)

    if(show == True):
        display(df.head())
        print_grafici(df, 'minute')

    return df


def tag_dataframe_by_time(data, timestamp_column, start_time, end_time, value, tag_column):
    
    start_ms = (start_time[0] * 60 * 1000) + (start_time[1] * 1000) + start_time[2]
    end_ms = (end_time[0] * 60 * 1000) + (end_time[1] * 1000) + end_time[2]
    
    if timestamp_column not in data.columns:
        raise ValueError(f"La colonna '{timestamp_column}' non esiste nei dati.")

    # Aggiungi il tag ai record che rientrano nel lasso di tempo
    data.loc[(data[timestamp_column] >= start_ms) & (data[timestamp_column] <= end_ms), tag_column] = value
    
    return data


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)

     # Matrice di confusione
    cm = confusion_matrix(test_labels, predictions, labels=[1,-1])
    tn, fp, fn, tp = cm.ravel() 

    #Calcolo metriche 
    accuracy = accuracy_score(test_labels, predictions)*100
    fbeta = fbeta_score(test_labels, predictions, beta=0.5, pos_label=-1)*100
    recall = recall_score(test_labels, predictions, pos_label=-1)*100
    f1 = f1_score(test_labels, predictions, pos_label=-1)*100
    precision = precision_score(test_labels, predictions, pos_label=-1)*100
        
    print('Model Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('Fbeta = {:0.2f}%.'.format(fbeta))
    print('Recall = {:0.2f}%.'.format(recall))
    print('F1 = {:0.2f}%.'.format(f1))
    print('Precision = {:0.2f}%.'.format(precision))

    print('-----------------------------------')
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print('-----------------------------------')
    
    return f1


def custom_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='binary')

def split_data_by_time(data, timestamp_column, split_time):
    
    split_time = (split_time[0] * 60 * 1000) + (split_time[1] * 1000) + split_time[2]
    
    if timestamp_column not in data.columns:
        raise ValueError(f"La colonna '{timestamp_column}' non esiste nei dati.")

    # Filtra i dati in base al tempo di split
    training_data = data[data[timestamp_column] <= split_time]
    test_data = data[data[timestamp_column] > split_time]

    return training_data, test_data

def split_scaler(file_path, time = (4, 0, 0)):
    df = pd.read_csv(file_path)

    columns_to_scale = ['GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'AngX', 'AngY', 'AngZ', 'AccX', 'AccY', 'AccZ']

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
    train_data, test_data = split_data_by_time(df_scaled, 'Timestamp', time)
    
        
    y_train = train_data['Tag']
    x_train = train_data[['AccX', 'AccY', 'AccZ', 'AngY', 'AngX']]
    
    y_test = test_data['Tag']
    x_test = test_data[['AccX', 'AccY', 'AccZ', 'AngY', 'AngX']]

    return x_train, y_train, x_test, y_test

def remove_data_in_range(data, timestamp_column, start_time, end_time):

    start_ms = (start_time[0] * 60 * 1000) + (start_time[1] * 1000) + start_time[2]
    end_ms = (end_time[0] * 60 * 1000) + (end_time[1] * 1000) + end_time[2]
    
    if timestamp_column not in data.columns:
        raise ValueError(f"La colonna '{timestamp_column}' non esiste nei dati.")

    # Filtrare i dati che non rientrano nel range
    filtered_df = data[(data[timestamp_column] < start_ms) | (data[timestamp_column] > end_ms)]

    return filtered_df