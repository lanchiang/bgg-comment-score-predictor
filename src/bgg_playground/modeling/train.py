import keras
import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, hp, Trials, fmin, tpe
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri(uri="http://localhost:8080")

data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";"
)

train, test = train_test_split(data, test_size=0.25, random_state=42)
train_x = train.drop(["quality"], axis=1).values
train_y = train[["quality"]].values.ravel()
test_x = test.drop(["quality"], axis=1).values
test_y = test[["quality"]].values.ravel()
train_x, valid_x, train_y, valid_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=42
)

signature = infer_signature(train_x, train_y)


def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y):
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)
    model = keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=mean, variance=var),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.Dense(units=1),
        ]
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=params["lr"], momentum=params["momentum"]
        ),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    with mlflow.start_run(nested=True):
        model.fit(
            x=train_x,
            y=train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=64
        )
        # Evaluate the model
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_rmse = eval_result[1]

        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)

        mlflow.tensorflow.log_model(model, "model", signature=signature)

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}


def objective(params):
    result = train_model(
        params,
        3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result


space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.0, 1.0),
}

mlflow.set_experiment("wine-quality-hyperopt")
with mlflow.start_run():
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=8,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]

    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=signature)

    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")
