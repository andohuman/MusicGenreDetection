from models import conv_gru_model
from utils import DataGenerator
import pandas as pd 
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import load_model, clone_model

TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
WORKERS = 20
MAX_QUEUE_SIZE = 20
EPOCHS = 50


save_file_path = 'models/model_{epoch:02d}_{val_acc:.2f}.hdf5'
chkpt = ModelCheckpoint(save_file_path, monitor='val_acc', save_best_only=False)
red_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.05, verbose=1, patience=3)
tensorboard = TensorBoard(log_dir='logs/', batch_size=TRAIN_BATCH_SIZE)

train_df = pd.read_csv('train_set.csv')
valid_df = pd.read_csv('valid_set.csv')

model = conv_gru_model(train_df['labels'].nunique())

#model = load_model('models/model_03_0.12.hdf5')


print(model.summary())

print('TRAINING ON ', len(train_df), ' DATA')
print('TESTING ON ', len(valid_df), ' DATA')

train_generator = DataGenerator(train_df, TRAIN_BATCH_SIZE, train_df['labels'].nunique())
validation_generator = DataGenerator(valid_df, VALID_BATCH_SIZE, train_df['labels'].nunique(), shuffle=False)

TRAIN_STEPS = int(len(train_df)/TRAIN_BATCH_SIZE)
VALID_STEPS = int(len(valid_df)/VALID_BATCH_SIZE)


model.fit_generator(generator=train_generator, \
					steps_per_epoch=TRAIN_STEPS, \
					epochs=EPOCHS, \
					use_multiprocessing=False, \
					shuffle=True, \
					validation_data = validation_generator,\
					callbacks=[chkpt, red_LR, tensorboard], \
					initial_epoch=0)
					#workers=WORKERS, \
					#max_queue_size=MAX_QUEUE_SIZE)





	

