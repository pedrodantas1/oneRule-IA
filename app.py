from collections import defaultdict

import numpy as np
import pandas as pd


class OneRClassifier:
    def __init__(self):
        self.best_rule = None
        self.best_feature = None
        self.min_error = float("inf")

    def _calculate_rule_error(self, feature_values, target_values):
        value_class_counts = defaultdict(lambda: defaultdict(int))

        # Conta ocorrências de cada classe para cada valor do atributo
        for value, target in zip(feature_values, target_values):
            value_class_counts[value][target] += 1

        # Para cada valor, escolhe a classe mais frequente
        rules = {}
        total_errors = 0

        for value in value_class_counts:
            # Encontra a classe majoritária
            counts = value_class_counts[value]
            majority_class = max(counts.items(), key=lambda x: x[1])[0]
            rules[value] = majority_class

            # Calcula erros para este valor
            total_value_instances = sum(counts.values())
            errors = total_value_instances - counts[majority_class]
            total_errors += errors

        return rules, total_errors

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Para cada feature, calcula as regras e erros
        for i in range(X.shape[1]):
            rules, errors = self._calculate_rule_error(X[:, i], y)

            if errors < self.min_error:
                self.min_error = errors
                self.best_rule = rules
                self.best_feature = i

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for instance in X:
            value = instance[self.best_feature]
            # Se encontrar um valor não visto no treino, usa a classe mais comum
            pred = self.best_rule.get(
                value,
                max(self.best_rule.values(), key=list(self.best_rule.values()).count),
            )
            predictions.append(pred)

        return np.array(predictions)


# Carregar dataset
def load_data():

    return pd.DataFrame(data)


# Demonstração do uso
if __name__ == "__main__":
    df = load_data()

    # Separa features e target
    X = df.drop("", axis=1)
    y = df[""]

    # Cria e treina o classificador
    classifier = OneRClassifier()
    classifier.fit(X, y)

    # Faz previsões
    predictions = classifier.predict(X)

    # Calcula acurácia
    accuracy = sum(predictions == y) / len(y)

    # Mostra resultados
    print(f"\nFeature mais relevante: {X.columns[classifier.best_feature]}")
    print("\nRegra encontrada:")
    for value, prediction in classifier.best_rule.items():
        print(
            f"Se {X.columns[classifier.best_feature]} = {value}, então credito = {prediction}"
        )
    print(f"\nAcurácia no conjunto de treino: {accuracy:.2%}")
