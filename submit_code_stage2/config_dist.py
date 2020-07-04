class DefaultConfigs(object):
	# print("config")
	train_data = "./dataset/train/" # where is your train data_dehaze
	test_data = "./dataset/test/"   # your test data_dehaze
	weights = "./checkpoints/"
	best_models = "./checkpoints/best_models/"
	submit = "./submit/"
	model_name = "seresnext5032x4d_resnet110-CE-03washing-v0-182v-192img-imgpre-addchusain0"#["bninception-bcelog", "resnet50-bcelog", "se_resnext50_32x4d-bcelog"]
	num_classes = 9
	img_weight = 192
	img_height = 192
	channels = 3
	lr = 0.0003
	batch_size = 64
	epochs = 15
	stage_epochs = [6, 2, 2, 1, 3]#[6, 3, 3, 3, 1, 7]#[6, 4, 5, 3, 7]#[6, 4, 5, 3, 1, 7]#[6, 4, 5, 3, 2, 6]
	weight_decay = 1e-4

config = DefaultConfigs()
