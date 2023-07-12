from comet_ml import Experiment
import os
import yaml
import struct
import pickle as pk
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.svm import SVC
from torch.utils import data

from g_model_CNN_c import raw_CNN_c

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.device(0)
torch.cuda.device(1)
torch.cuda.get_device_name(0)

def get_utt_list(src_dir):
	'''
	Designed for DCASE2019 task 1-a
	'''
	l_utt = []
	for r, ds, fs in os.walk(src_dir):
		for f in fs:
			if f[-3:] != 'npy':
				continue
			k = f.split('.')[0]
			l_utt.append(k)

	return l_utt

def mixup_data(x, y, alpha=1.0, use_cuda=True):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CenterLoss(nn.Module):
	"""Center loss.
	
	Reference:
	Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
	
	Args:
		num_classes (int): number of classes.
		feat_dim (int): feature dimension.
	"""
	def __init__(self, num_classes = None, feat_dim = None, use_gpu = True, device = None):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.use_gpu = use_gpu
		self.device = device

		if self.use_gpu:
			#self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
		else:
			self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

	def forward(self, x, labels):
		"""
		Args:
			x: feature matrix with shape (batch_size, feat_dim).
			labels: ground truth labels with shape (batch_size).
		"""
		batch_size = x.size(0)
		distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
				  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
		distmat.addmm_(1, -2, x, self.centers.t())

		classes = torch.arange(self.num_classes).long()
		#if self.use_gpu: classes = classes.cuda()
		if self.use_gpu: classes = classes.to(self.device)
		labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
		mask = labels.eq(classes.expand(batch_size, self.num_classes))

		dist = distmat * mask.float()
		loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

		return loss

#Custom summary method
def summary(model, input_size, batch_size=-1, device="cuda", print_fn = None):
	if print_fn == None: printfn = print

	def register_hook(module): #Hook is used to update gradients

		def hook(module, input, output):
#Hook retrieves the class name, module key and index then sets the input shape of that module index to the size of input
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			module_idx = len(summary)

			m_key = "%s-%i" % (class_name, module_idx + 1)
			summary[m_key] = OrderedDict()
			summary[m_key]["input_shape"] = list(input[0].size())
			summary[m_key]["input_shape"][0] = batch_size
			if isinstance(output, (list, tuple)):
				summary[m_key]["output_shape"] = [
					[-1] + list(o.size())[1:] for o in output
				]
			else:
				summary[m_key]["output_shape"] = list(output.size())
				summary[m_key]["output_shape"][0] = batch_size

			params = 0
			if hasattr(module, "weight") and hasattr(module.weight, "size"):
				params += torch.prod(torch.LongTensor(list(module.weight.size())))
				summary[m_key]["trainable"] = module.weight.requires_grad
			if hasattr(module, "bias") and hasattr(module.bias, "size"):
				params += torch.prod(torch.LongTensor(list(module.bias.size())))
			summary[m_key]["nb_params"] = params

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
			and not (module == model)
		):
			hooks.append(module.register_forward_hook(hook))

	device = device.lower()
	#'''
	assert device in [
		"cuda",
		"cpu",
	], "Input device is not valid, please specify 'cuda' or 'cpu'"
	#'''

	#dtype = torch.cuda.FloatTensor
	#'''
	if device == "cuda" and torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor
	#'''

	# multiple inputs to the network
	if isinstance(input_size, tuple):
		input_size = [input_size]

	# batch_size of 2 for batchnorm
	x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
	# print(type(x[0]))

	# create properties
	summary = OrderedDict()
	hooks = []

	# register hook
	model.apply(register_hook)

	# make a forward pass
	# print(x.shape)
	model(*x)

	# remove these hooks
	for h in hooks:
		h.remove()

	print_fn("----------------------------------------------------------------")
	line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
	print_fn(line_new)
	print_fn("================================================================")
	total_params = 0
	total_output = 0
	trainable_params = 0
	for layer in summary:
		# input_shape, output_shape, trainable, nb_params
		line_new = "{:>20}  {:>25} {:>15}".format(
			layer,
			str(summary[layer]["output_shape"]),
			"{0:,}".format(summary[layer]["nb_params"]),
		)
		total_params += summary[layer]["nb_params"]
		total_output += np.prod(summary[layer]["output_shape"])
		if "trainable" in summary[layer]:
			if summary[layer]["trainable"] == True:
				trainable_params += summary[layer]["nb_params"]
		print_fn(line_new)

	# assume 4 bytes/number (float on cuda).
	total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
	total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
	total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
	total_size = total_params_size + total_output_size + total_input_size

	print_fn("================================================================")
	print_fn("Total params: {0:,}".format(total_params))
	print_fn("Trainable params: {0:,}".format(trainable_params))
	print_fn("Non-trainable params: {0:,}".format(total_params - trainable_params))
	print_fn("----------------------------------------------------------------")
	print_fn("Input size (MB): %0.2f" % total_input_size)
	print_fn("Forward/backward pass size (MB): %0.2f" % total_output_size)
	print_fn("Params size (MB): %0.2f" % total_params_size)
	print_fn("Estimated Total Size (MB): %0.2f" % total_size)
	print_fn("----------------------------------------------------------------")
			

class Dataset_DCASE2019_t1(data.Dataset):
	#def __init__(self, list_IDs, labels, nb_time, base_dir):
	def __init__(self, lines, d_class_ans, nb_samp, cut, base_dir):
		'''
		self.lines		: list of strings 
		'''
		self.lines = lines 
		self.d_class_ans = d_class_ans 
		self.base_dir = base_dir
		self.nb_samp = nb_samp
		self.cut = cut

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, index):
		k = self.lines[index] # k = filename['index']
		X = self.pre_emp_multiChannel(np.load(self.base_dir+k+'.npy')) # x is the pre-emphasis augmented k
		y = self.d_class_ans[k.split('-')[0]] # extract class from filename
		n_channels, n_samples = X.shape
		# if n_samples > 480000:
		# 	X=X[:,:480000]
		# 	# print(f'Truncated to{X.shape}')
		# if n_samples ==479999:
		# 	X=np.pad(X,((0,0),(0,1)),'constant')
		# 	# print(f'Padded to:{X.shape}')
		# if n_samples ==479998:
		# 	X=np.pad(X,((0,0),(0,2)),'constant')
		# if not X.shape == (2,480000):
		# 	print(f'ERROR: I messed up:{X.shape}')
		if self.cut:
			nb_samp = X.shape[1]
			# print(nb_samp)
			# start_idx = 0
			start_idx = np.random.randint(low = 0, high = nb_samp - self.nb_samp)
			# print(start_idx)
			# print(start_idx+self.nb_samp)
			X = X[:, start_idx:start_idx+self.nb_samp]
			# X=X[:,start_idx:480000]
			# print(start_idx+self.nb_samp)
			# print(f'Final Shape:{X.shape}')
		# else: X = X[:, :480000]	
		else: X = X[:, :479520]
		# else: X = X[:, :479999]
		X *= 32000
		# print(X)
		return X, y
	#Pre-emphasis augmentation for raw waveforms only
	def pre_emp_multiChannel(self, x):
		'''
		input	: (#channel, #time)	
		output	: (#channel, #time)		
		'''
		return np.asarray(x[:, 1:] - 0.97 * x[:, :-1], dtype=np.float32)
#Create dictionary to store corresponding labels by stripping and splititng meta csv entries
def make_labeldic(lines):
	idx = 0
	dic_label = {}
	list_label = []
	for line in lines:
		label = line.strip().split('/')[1].split('-')[0] # Kenneth: I think this should be left as forward slash because the lines in the csv meta files are using forward slashes
		if label not in dic_label:
			dic_label[label] = idx
			list_label.append(label)
			idx += 1
	return (dic_label, list_label)

# split dataset into training and validation sets # lines refer to filenames from csv rows
def split_dcase2019_fold(fold_scp, lines):
	fold_lines = open(fold_scp, 'r').readlines() #Reads fold1_train.csv
	dev_lines = []
	val_lines = []

	fold_list = []
	for line in fold_lines[1:]:
		fold_list.append(line.strip().split('\t')[0].split('/')[1].split('.')[0]) # Kenneth: I think this should be left as forward slash because the lines in the csv meta files are using forward slashes
		
	for line in lines:
		if line in fold_list:
			dev_lines.append(line)
		else:
			val_lines.append(line)

	return dev_lines, val_lines

if __name__ == '__main__':
	#load yaml file & set comet_ml config
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml, Loader=yaml.FullLoader) # Kenneth: See https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation. Your yaml version is too new compared to the authors'.
                                                           # Kenneth: How did I get here?
                                                           # Kenneth: First, I just Googled the error: https://www.google.com/search?q=TypeError%3A+load%28%29+missing+1+required+positional+argument%3A+%27Loader%27&rlz=1C1GCEU_enSG1018SG1018&sxsrf=APwXEddUNwh6wAs0u8XS7Sf5ABh-rY5aMQ%3A1687959497913&ei=yTecZK-tN_S84-EPpdCsoAY&ved=0ahUKEwjvtZKSi-b_AhV03jgGHSUoC2QQ4dUDCA8&uact=5&oq=TypeError%3A+load%28%29+missing+1+required+positional+argument%3A+%27Loader%27&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwAzIKCAAQRxDWBBCwA0oECEEYAFDTAljTAmC4BWgBcAF4AIABAIgBAJIBAJgBAKABAqABAcABAcgBCA&sclient=gws-wiz-serp
                                                           # Kenneth: The first link already gives the solution: https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col
                                                           # Kenneth: The top rated reply is "Now, the load() function requires parameter loader=Loader."
                                                           # Kenneth: Which hints that it's a version problem.
                                                           # Kenneth: The answer also gives the helpful link at https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
                                                           # Kenneth: Which now _explicitly_ states that it's a version problem.
                                                           # Kenneth: Then, I just ran yaml.__version__ on your environment and found that it's 6.0.
                                                           # Kenneth: Obviously, that's more than 5.1. QED.
                                                           # Kenneth: My advice for the future is to read the results critically and carefully so that you don't miss small details. Especially since the first Google search result solves your problem already.
	experiment = Experiment(api_key="X8sW8l8RL3gPZt0ZKyGz7Swek",
		project_name="specialist-kd", workspace="seanyeo300",
		auto_output_logging = 'simple',
		disabled = bool(parser['comet_disable']))
	experiment.set_name(parser['name'])
	
	#device setting
	cuda = torch.cuda.is_available()
	device = torch.device('cuda:%s'%parser['gpu_idx'][0] if cuda else 'cpu')

	#get DB list
	lines = get_utt_list(parser['DB']+'wave_np')

	#get label dictionary
	if bool(parser['make_label_dic']):
		with open(parser['DB']+parser['meta_scp']) as f:
			l_meta = f.readlines()
		d_class_ans, l_class_ans  = make_labeldic(l_meta[1:])
		pk.dump([d_class_ans, l_class_ans], open(parser['DB']+parser['dir_label_dic'], 'wb'))
	else:
		d_class_ans, l_class_ans = pk.load(open(parser['DB']+parser['dir_label_dic'], 'rb'))

	#split trnset and devset
	trn_lines, dev_lines = split_dcase2019_fold(fold_scp = parser['DB']+parser['fold_scp'], lines = lines)
	print(len(trn_lines), len(dev_lines))
	del lines
	if bool(parser['comet_disable']):
		np.random.shuffle(trn_lines)
		np.random.shuffle(dev_lines)
		trn_lines = trn_lines[:1000]
		dev_lines = dev_lines[:1000]

	#define dataset generators
	# Kenneth: print(f"\tNUMBER SAMPLES = {parser['nb_samp']}")
	# Kenneth: print(f"\ttype(trn_lines) = list, length = {len(trn_lines)}")
	# Kenneth: print(f"\ttype(d_class_ans) = dict, keys = {d_class_ans.keys()}")
	# Kenneth: print(f"\tbase_dir = {parser['DB']+parser['wav_dir']}")
	trnset = Dataset_DCASE2019_t1(lines = trn_lines,
		d_class_ans = d_class_ans,
		nb_samp = parser['nb_samp'],
		cut = True,
		base_dir = parser['DB']+parser['wav_dir'])
    # Kenneth: Reminder, data is from torch.utils
	trnset_gen = data.DataLoader(trnset,
		batch_size = parser['batch_size'],
		shuffle = True,
		num_workers = parser['nb_proc_db'],
		drop_last = True) #drops the last incomplete batch
	devset = Dataset_DCASE2019_t1(lines = dev_lines,
		d_class_ans = d_class_ans,
		nb_samp = 0,
		cut = False,
		base_dir = parser['DB']+parser['wav_dir'])
	devset_gen = data.DataLoader(devset,
		batch_size = parser['batch_size'],
		shuffle = False,
		num_workers = parser['nb_proc_db'],
		drop_last = False)

	#set save directory
    # Kenneth: All file directories must be in Windows format to work on your PC.
    # Kenneth: The authors used Mac/Linux (and so does everyone except us), so their directories will give errors if running on a Windows PC.
    # Kenneth: As a rule of thumb, always assume all ML code is built on Linux unless otherwise stated by the authors.
    # Kenneth: Converting it to Windows is normally assumed to be the responsibility of Windows users, since they are by far the minority in the ML scene.
	save_dir = parser['save_dir'] + parser['name'] + '\\'
	print(f"\tSAVE DIRECTORY IS {save_dir}") # Kenneth: I added this line to confirm where the logs & model files will be saved to.
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if not os.path.exists(save_dir  + 'results\\'):
		os.makedirs(save_dir + 'results\\')
	if not os.path.exists(save_dir  + 'weights\\'):
		os.makedirs(save_dir + 'weights\\')
	if not os.path.exists(save_dir  + 'svm\\'):
		os.makedirs(save_dir + 'svm\\')
	
	#log experiment parameters to local and comet_ml server
	#to local
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.close()

	#to comet server
	experiment.log_parameters(parser)
	experiment.log_parameters(parser['model'])

	#define model
	model = raw_CNN_c(parser['model']).to(device)

	#log model summary to file
	with open(save_dir + 'summary.txt', 'w+') as f_summary:
		#summ = summary(model, input_size = (parser['model']['in_channels'], parser['nb_time'], parser['feat_dim'])) # configure input_size as (channels, H, W)
		summary(model,
			input_size = (parser['model']['in_channels'], parser['nb_samp']), #from yaml file: in_channels = 2, nb_sampl = 240000 == 480000/2
			print_fn=lambda x: f_summary.write(x + '\n')) 

	if len(parser['gpu_idx']) > 1:
		model = nn.DataParallel(model, device_ids = parser['gpu_idx'])

	#set ojbective funtions
	criterion = nn.CrossEntropyLoss()
	c_obj_fn = CenterLoss(num_classes = parser['model']['nb_classes'],
		feat_dim = parser['model']['nb_fc_node'],
		device = device)

	#set optimizer
	params = list(model.parameters()) + list(c_obj_fn.parameters())
	if parser['optimizer'].lower() == 'sgd':
		optimizer = torch.optim.SGD(params,
			lr = parser['lr'],
			momentum = parser['opt_mom'],
			weight_decay = parser['wd'], #weight decay 0.001
			nesterov = bool(parser['nesterov']))

	elif parser['optimizer'].lower() == 'adam':
		#optimizer = torch.optim.Adam(params,
		optimizer = torch.optim.Adam(model.parameters(),
			lr = parser['lr'],
			weight_decay = parser['wd'], #weight decay 0.001
			amsgrad = bool(parser['amsgrad']))
	'''
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
		milestones = parser['lrdec_milestones'],
		gamma = parser['lrdec'])
	'''
	##########################################
	#train/val################################
	##########################################
	best_acc = 0.
	f_acc = open(save_dir + 'accs.txt', 'a', buffering = 1)
	for epoch in tqdm(range(parser['epoch'])):
		#train phase
		model.train()
		mixup = True if epoch > parser['mixup_start'] else False #mixup starts after N epochs. In this case, 5
		with tqdm(total = len(trnset_gen), ncols = 70) as pbar:
			for m_batch, m_label in trnset_gen:
				#continue #temporary
				m_batch, m_label = m_batch.to(device), m_label.to(device)
				
				#mixup data if met condition
				if mixup:
					m_batch, m_label_a, m_label_b, lam = mixup_data(m_batch, m_label,
						alpha = parser['mixup_alpha'],
						use_cuda = True)
					m_batch, m_label_a, m_label_b = map(torch.autograd.Variable, [m_batch, m_label_a, m_label_b])
					code, output = model(m_batch)
					 #Mixup criterion defined in line 52
					cce_loss = mixup_criterion(criterion, output, m_label_a, m_label_b, lam)
					c_loss = mixup_criterion(c_obj_fn, code, m_label_a, m_label_b, lam)
				else:
					code, output = model(m_batch) # Model returns x,y == code,output
					cce_loss = criterion(output, m_label) #CCE loss calculated on output of ResNet (criterion was set as CCE)
					c_loss = c_obj_fn(code, m_label) #c_obj_fn set as class(center_loss)
				loss = cce_loss + (parser['c_loss_weight'] * c_loss) #Calculate total loss

				#print(loss)
				optimizer.zero_grad() #Optimizer set as either SGD or ADAM # default param set_to_none = True
				# loss.zero_grad() clears x.grad for every parameter x in the optimizer. It’s important to call this before loss.backward(), otherwise you’ll accumulate the gradients from multiple passes.
				loss.backward()
				for param in c_obj_fn.parameters():
					param.grad.data *= (parser['c_loss_lr'] / (parser['c_loss_weight'] * parser['lr']))
				optimizer.step()
				pbar.set_description('epoch: %d loss: %.3f'%(epoch, loss))
				pbar.update(1)
		experiment.log_metric('trn_loss', loss, step=epoch)
		#lr_scheduler.step()

		#validation phase
		model.eval()
		with torch.set_grad_enabled(False):
			embeddings_dev = []
			data_y_dev = []
			with tqdm(total = len(devset_gen), ncols = 70) as pbar:
				for m_batch, m_label in devset_gen:
					m_batch = m_batch.to(device)
					# code,_ = x,y from the return of model(m_batch)
					code, _ = model(m_batch)
					m_label = list(m_label.numpy())
					embeddings_dev.extend(list(code.cpu().numpy())) #>>> (16, 64?)
					data_y_dev.extend(m_label)
					pbar.set_description('epoch%d: Extract ValEmbeddings'%(epoch))
					pbar.update(1)
			embeddings_dev = np.asarray(embeddings_dev, dtype = np.float32)
			print(embeddings_dev.shape)

			embeddings_trn = []
			data_y = []
			with tqdm(total = len(trnset_gen), ncols = 70) as pbar:
				for m_batch, m_label in trnset_gen:
					m_batch = m_batch.to(device)
					code, _ = model(m_batch)
					m_label = list(m_label.numpy())
					embeddings_trn.extend(list(code.cpu().numpy())) #>>> (16, 64?)
					data_y.extend(m_label)
					pbar.set_description('epoch%d: Extract TrnEmbeddings'%(epoch))
					pbar.update(1)
			embeddings_trn = np.asarray(embeddings_trn, dtype = np.float32)
			
			SVM_list = []
			acc = []
			classwise_acc = []
			for cov_type in ['rbf', 'sigmoid']:
				score_list = []
		
				SVM_list.append(SVC(kernel=cov_type,
					gamma = 'scale',
					probability = True))
				SVM_list[-1].fit(embeddings_trn, data_y)
		
				num_corr = 0
				num_corr_class = [0]* len(l_class_ans)
				num_predict_class = [0] * len(l_class_ans)
		
				score_list = SVM_list[-1].predict(embeddings_dev)
				
				assert len(score_list) == len(data_y_dev)
				for i in range(embeddings_dev.shape[0]):
					num_predict_class[score_list[i]] += 1
					if score_list[i] == data_y_dev[i]:
						num_corr += 1
						num_corr_class[data_y_dev[i]] += 1
				acc.append(float(num_corr)/ embeddings_dev.shape[0])
				classwise_acc.append(np.array(num_corr_class) / np.array(num_predict_class))
				print(classwise_acc[-1], acc[-1])
			f_acc.write('%d %f %f\n'%(epoch, float(acc[0]), float(acc[1])))
	
			max_acc = max(acc[0], acc[1])
			experiment.log_metric('val_acc_rbf', acc[0],step=epoch)
			experiment.log_metric('val_acc_sig', acc[1],step=epoch)
			#record best validation model
			if max_acc > best_acc:
				print('New best acc: %f'%float(max_acc))
				best_acc = float(max_acc)
				experiment.log_metric('best_val_acc', best_acc,step=epoch)
				
				#save best model
				if acc[0] > acc[1]:
					pk.dump((SVM_list[0], classwise_acc[0]), open(save_dir + 'svm\\best_rbf.pk', 'wb'))
					if len(parser['gpu_idx']) > 1: # multi GPUs
						torch.save(model.module.state_dict(), save_dir +  'weights\\best_rbf.pt')
					else: #single GPU
						torch.save(model.state_dict(), save_dir +  'weights\\best_rbf.pt')
				else:
					pk.dump((SVM_list[1], classwise_acc[1]), open(save_dir + 'svm\\best_sig.pk', 'wb'))
					if len(parser['gpu_idx']) > 1: # multi GPUs
						torch.save(model.module.state_dict(), save_dir +  'weights\\best_sig.pt')
					else: #single GPU
						torch.save(model.state_dict(), save_dir +  'weights\\best_sig.pt')
				
	f_acc.close()








