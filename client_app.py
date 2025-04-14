from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from random import random
import numpy as np

from fed.task import (
    IncomeClassifier,
    evaluate,
    get_weights,
    load_data,
    set_weights,
    train,
    eval_learning
)


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, X_train, X_test, y_train, y_test, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        weights = get_weights(self.net)
        #if(self.partition_id == 1):
           # np.asarray(weights)
           # weights.fill(0.0)
            #weights = weights.tolist()
            #weights.fill(0.0)
                #print("\n\n\n\nWeights:\n\n\n")
                #print(weights)
        return weights, len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, y_pred = evaluate(self.net, self.testloader)

        acc, rec, prec, f1 = eval_learning(self.y_test, y_pred)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
        }
        return loss, len(self.testloader), output_dict#{"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader, X_train, X_test, y_train, y_test = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    net = IncomeClassifier()
    return FlowerClient(net, train_loader, test_loader, X_train, X_test, y_train, y_test, partition_id).to_client()


app = ClientApp(client_fn=client_fn)