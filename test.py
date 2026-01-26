

reducer = AVLearner()
train_y = reducer.fit_transform(embedding, label, init_global)
test_y = reducer.transform(test_embedding)