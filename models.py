from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Reshape, Input, Concatenate, AveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D, \
						BatchNormalization, MaxPooling2D, Activation, Dropout, CuDNNGRU, ELU
from keras.optimizers import RMSprop
from keras.models import Model


def conv_gru_model(n_classes):

	feat_dict = {'mel_input':Input(shape=(128, 1292, 1)), \
	'chroma_input':Input(shape=(12, 1292, 1)), \
	'mfcc_input':Input(shape=(20, 1292, 1)), \
	'spec_input':Input(shape=(7, 1292, 1)), \
	'tonnetz_input':Input(shape=(6, 1292, 1)),\

	'mel':None, \
	'chroma':None, \
	'mfcc':None, \
	'spec':None, \
	'tonnetz':None}

	sub_models = ['mel', 'chroma', 'mfcc', 'spec', 'tonnetz']
	#sub_models = ['mel']
	for sub_model in sub_models:

		if sub_model is 'mel': pool_size = [2,(3,2),(3,2),(4,3)]
		elif sub_model is 'chroma': pool_size = [(2,2),(2,2),(2,2),(1,3)]
		elif sub_model is 'mfcc': pool_size = [(2,2),(2,2),(2,2),(2,3)]
		else: pool_size = [(2,2),(2,2),(1,2),(1,3)]

		#CONV BLOCK 1
		feat_dict[sub_model] = Conv2D(64, 3, padding='same', name=sub_model+'_conv1')(feat_dict[sub_model+'_input'])
		feat_dict[sub_model] = BatchNormalization(name=sub_model+'_bn1')(feat_dict[sub_model])
		feat_dict[sub_model] = ELU(name=sub_model+'_activation1')(feat_dict[sub_model])
		feat_dict[sub_model] = MaxPooling2D(pool_size=pool_size[0],name=sub_model+'_maxpool1')(feat_dict[sub_model])
		feat_dict[sub_model] = Dropout(0.25, name=sub_model+'_dropout1')(feat_dict[sub_model])

		#CONV BLOCK 2
		feat_dict[sub_model] = Conv2D(128, 3, padding='same', name=sub_model+'_conv2')(feat_dict[sub_model])
		feat_dict[sub_model] = BatchNormalization(name=sub_model+'_bn2')(feat_dict[sub_model])
		feat_dict[sub_model] = ELU(name=sub_model+'_activation2')(feat_dict[sub_model])
		feat_dict[sub_model] = MaxPooling2D(pool_size=pool_size[1],name=sub_model+'_maxpool2')(feat_dict[sub_model])
		feat_dict[sub_model] = Dropout(0.25, name=sub_model+'_dropout2')(feat_dict[sub_model])

		#CONV BLOCK 3
		feat_dict[sub_model] = Conv2D(128, 3, padding='same', name=sub_model+'_conv3')(feat_dict[sub_model])
		feat_dict[sub_model] = BatchNormalization(name=sub_model+'_bn3')(feat_dict[sub_model])
		feat_dict[sub_model] = ELU(name=sub_model+'_activation3')(feat_dict[sub_model])
		feat_dict[sub_model] = MaxPooling2D(pool_size=pool_size[2],name=sub_model+'_maxpool3')(feat_dict[sub_model])
		feat_dict[sub_model] = Dropout(0.25, name=sub_model+'_dropout3')(feat_dict[sub_model])

		#CONV BLOCK 4
		feat_dict[sub_model] = Conv2D(256, 3, padding='same', name=sub_model+'_conv4')(feat_dict[sub_model])
		feat_dict[sub_model] = BatchNormalization(name=sub_model+'_bn4')(feat_dict[sub_model])
		feat_dict[sub_model] = ELU(name=sub_model+'_activation4')(feat_dict[sub_model])
		feat_dict[sub_model] = MaxPooling2D(pool_size=pool_size[3],name=sub_model+'_maxpool4')(feat_dict[sub_model])
		feat_dict[sub_model] = Dropout(0.25, name=sub_model+'_dropout4')(feat_dict[sub_model])

		feat_dict[sub_model] = Reshape((53, 256))(feat_dict[sub_model])

		#GRU BLOCK

		feat_dict[sub_model] = CuDNNGRU(128, return_sequences=True, name=sub_model+'_gru1')(feat_dict[sub_model])
		feat_dict[sub_model] = CuDNNGRU(128, return_sequences=False,name=sub_model+'_gru2')(feat_dict[sub_model])

	x = Concatenate()([feat_dict[i] for i in sub_models])
	x = Dropout(0.3)(x)
	x = Dense(256, activation='elu')(x)
	x = Dropout(0.5)(x)
	pred = Dense(n_classes, activation='softmax')(x)

	#model = Model(inputs=feat_dict['mel_input'], outputs=feat_dict['mel'])
	model = Model(inputs=[feat_dict[i+'_input'] for i in sub_models], outputs=pred)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	return model





'''
def conv_lstm_model():

	sub_model_dict = {'mel_input':Input(shape=(1292, 128)), \
	'chroma_input':Input(shape=(1292, 12)), \
	'mfcc_input':Input(shape=(1292, 20)), \
	'spec_input':Input(shape=(1292, 7)), \
	'tonnetz_input':Input(shape=(1292, 6)),\

	'mel':None, \
	'chroma':None, \
	'mfcc':None, \
	'spec':None, \
	'tonnetz':None}

	for feat in ['mel', 'chroma', 'mfcc', 'spec', 'tonnetz']:

		sub_model_dict[feat] = Conv1D(filters=128, kernel_size=4)(sub_model_dict[feat+'_input'])
		sub_model_dict[feat] = BatchNormalization()(sub_model_dict[feat])
		sub_model_dict[feat] = Activation('elu')(sub_model_dict[feat])
		sub_model_dict[feat] = MaxPooling1D()(sub_model_dict[feat])
		sub_model_dict[feat] = Dropout(0.5)(sub_model_dict[feat])

		sub_model_dict[feat] = Conv1D(filters=256, kernel_size=4)(sub_model_dict[feat])
		sub_model_dict[feat] = BatchNormalization()(sub_model_dict[feat])
		sub_model_dict[feat] = Activation('elu')(sub_model_dict[feat])
		sub_model_dict[feat] = MaxPooling1D()(sub_model_dict[feat])
		sub_model_dict[feat] = Dropout(0.5)(sub_model_dict[feat])

		#sub_model_dict[feat] = CuDNNLSTM(256, return_sequences=True)(sub_model_dict[feat])

		sub_model_dict[feat] = Reshape((320, 256, 1))(sub_model_dict[feat])



	x = Concatenate(axis=-1)([sub_model_dict['mel'],\
		sub_model_dict['chroma'],\
		sub_model_dict['mfcc'],\
		sub_model_dict['spec'],\
		sub_model_dict['tonnetz']])

	x = mobnet_remix()(x)

	#x = Dense(512, activation='elu')(x)

	pred = Dense(8, activation='softmax')(x)

	model = Model(inputs=[sub_model_dict['mel_input'], \
		sub_model_dict['chroma_input'], \
		sub_model_dict['mfcc_input'], \
		sub_model_dict['spec_input'], \
		sub_model_dict['tonnetz_input']], \
		outputs=pred)
	
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )

	return model

'''
