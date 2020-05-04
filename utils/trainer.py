import torch.nn as nn
import torch


class ModelTrainer():
    def __init__(self, model, params):
        self.model=model
        self.params=params
        self.optimizer = self._init_optimizer()
        self.loss = self._init_loss()


    # TODO:
    # should contain: Loss, optimizer, train method, validation methods
    def _init_optimizer(self):
        if self.params.optimizer_type == 'adam':
            return torch.optim.Adam(parameters=self.model.parameters(), lr=self.params.learning_rate[0])
            # optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate[0], clipnorm=params.gradient_clip_th)
        elif self.params.optimizer_type == 'sgd':
            return torch.optim.SGD(lr=self.params.learning_rate[0])
        else:
            raise Exception('optimizer_type not supported: ' + self.params.optimizer_type)

    def _init_loss(self):
        return


    def train(self, dataloader):
        self.model.train()
        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            states, preds = self.model(inputs) # TODO: choose only inputs from batch
            loss = self.loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.gradient_clip_th)
            self.optimizer.step()





