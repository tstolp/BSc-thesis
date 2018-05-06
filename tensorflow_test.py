import tensorlfow as tf 

'''
shape of X_train:(8756, 6)
shape of X_test: (3213, 6)
shape of Y_train: (8756, 5)
shape of Y_test: (3213, 5)
'''
n_nodes_hl1 = 7

n_classes = 5

x = tf.placeholder('float',[None,6])
y = tf.placeholder('float')

def neural_network_model(data):

	# (input_data * weights) + biases 

	hidden_1_layer = {'weights':tf.variable(tf.random_normal([6, n_nodes_hl1])),
	'biases':tf.variable(tf.random_normal([n_nodes_hl1]))}

	output_layer = {'weights':tf.variable(tf.random_normal([n_nodes_hl1, n_classes])),
	'biases':tf.variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) # activation function
	
	output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x)
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backpropagation
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range():
				epoch_x, epoch_y = 
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of' hm_epochs, 'loss:', epoch_loss)


		correct = tf. equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x: , y:}))