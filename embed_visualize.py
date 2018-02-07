import tensorflow as tf
import dataset_factory
import pixelda_model
from hparams import create_hparams
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
import pickle

import sklearn
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir','./input_mask_contrast_no_content_entropy',
                    'Model saved directory')
flags.DEFINE_string('log_dir','./embed','directory to save log')
flags.DEFINE_integer('num_readers',4,
                     'The number of parallel readers that read data from the dataset')
flags.DEFINE_integer('num_preprocessing_threads',4,
                     'The number of threads used to create the batches.')
flags.DEFINE_string('mode','marginal','marginal/lateral/head')
flags.DEFINE_integer('filenum',1000,'')
flags.DEFINE_integer('RS', 28972, 'random seed')
flags.DEFINE_integer('embed_dim', 2, 'embedding dimension')

if FLAGS.mode == 'marginal':
    num_class = 3
#    total_num_class = 9
else:
    num_class = 3
#    total_num_class = 6

def scatter3D(x, colors, embed_type='tsne'):
	# Choosing color, make numpy array
    palette = np.array(sns.color_palette("hls",num_class))
  
    fig = plt.figure(figsize=(16,16))
    #ax = plt.subplot(aspect='equal')
    ax = fig.add_subplot(111,projection='3d')
    labels = ['source','transferred','target']
    # For 3 dimension
    for i in range(num_class):  
        ax.scatter(x[colors==i, 0], x[colors==i ,1], x[colors==i ,2],lw=0, s=40, c=palette[i], label= labels[i])
  
    plt.xlim(-13,13)
    plt.ylim(-13,13)
    ax.axis('off')
    ax.axis('tight')
    legend = ax.legend(fontsize=30,frameon=True,bbox_to_anchor=(1,1))
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    #ax.legend(fontsize=45)
    #ax.legend(frameon=True)

    class_list = ['S','F','T']
    txts = []
    for i in range(num_class):
        xtext, ytext, ztext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, ztext, class_list[i], fontsize=12)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()])
        txts.append(txt)

    if embed_type == 'tsne':
        plt.savefig(os.path.join(FLAGS.log_dir + '/tsne3d.png'), dpi=120)    
    elif embed_type == 'kpca':
        plt.savefig(os.path.join(FLAGS.log_dir + '/kpca3d.png'), dpi=120)
    else:
        raise NotImplementedError
    plt.close()

    return fig, ax, txts

def scatter2D(x, colors, embed_type='tsne'):
    palette = np.array(sns.color_palette("hls",num_class))
 
    f = plt.figure(figsize=(16,16))
    ax = plt.subplot(aspect='equal')
    if FLAGS.mode == 'marginal':
        labels = ['source', 'transferred', 'target']
        marker = ['^','o','*']
    else:
        labels = ['trm','trn','trp','tgm','tgn','tgp']
        marker = ['^','o','*','^','o','*']
    # For 2 dimension
    for i in range(num_class):
        sc = ax.scatter(x[colors==i, 0], x[colors==i ,1], marker=marker[i], lw =0 , s=160, c=palette[i], label= labels[i])
  
    plt.xlim(-13,13)
    plt.ylim(-13,13)
    ax.axis('off')
    ax.axis('tight')
    legend = ax.legend(fontsize=30,frameon=True,bbox_to_anchor=(1,1))
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    #ax.legend(fontsize=45)
    #ax.legend(frameon=True)

    class_list = ['S','F','T']
    txts = []
    for i in range(num_class):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, class_list[i], fontsize=30, fontweight='bold')
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground='w'),
            PathEffects.Normal()])
        txts.append(txt)

    if embed_type == 'tsne':
        plt.savefig(os.path.join(FLAGS.log_dir + '/tsne2d.pdf'), dpi=120)    
    elif embed_type == 'kpca':
        plt.savefig(os.path.join(FLAGS.log_dir + '/kpca2d_linear.pdf'), dpi=120)
    elif embed_type == 'iso':
        plt.savefig(os.path.join(FLAGS.log_dir + '/iso2d.pdf'), dpi=120)
    else:
        raise NotImplementedError

    plt.close()

    return f, ax, sc, txts

def visualize(checkpoint_dir,hparams,log_dir):
    hparams.batch_size = 10
    target_images, target_labels = dataset_factory.provide_batch(
        'target','train','data_target',
        FLAGS.num_readers, hparams.batch_size,
        FLAGS.num_preprocessing_threads)
    source_images, source_labels = dataset_factory.provide_batch(
        'source','train','data_source',
        FLAGS.num_readers, hparams.batch_size,
        FLAGS.num_preprocessing_threads)
    source_labels = tf.argmax(source_labels['classes'],1)
    target_labels = tf.argmax(target_labels['classes'],1)
 
    mask_images = source_images[:,:,:,3]
    source_images = source_images[:,:,:,:3]
    if hparams.input_mask:
        mask_images = tf.to_float(tf.greater(mask_images,0.99))
        source_images = tf.multiply(source_images,tf.tile(tf.expand_dims(mask_images,3),[1,1,1,3]))
      
    end_points = pixelda_model.create_model(
        hparams, target_images, target_images,
        source_images=source_images,
        is_training=False, noise=None,
        num_classes=3)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #im_num = 120
    source_im_array      = np.zeros((FLAGS.filenum,90,160,3))
    transferred_im_array = np.zeros((FLAGS.filenum,90,160,3))
    target_im_array      = np.zeros((FLAGS.filenum,90,160,3)) 

    source_lb_array      = np.zeros(FLAGS.filenum)
    transferred_lb_array = np.zeros(FLAGS.filenum)
    target_lb_array      = np.zeros(FLAGS.filenum)
  
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,os.path.join(FLAGS.checkpoint_dir,ckpt.model_checkpoint_path.split('/')[-1]))
    
        for i in range(FLAGS.filenum//hparams.batch_size):
            s_im,s_lb,tr_im,tg_im,tg_lb = sess.run([source_images,source_labels,end_points['transferred_images'],target_images,target_labels])
            s_lateral_lb = (s_lb % 9 )/3
            s_head_lb    = s_lb % 3
            source_im_array[i*hparams.batch_size:(i+1)*hparams.batch_size,:,:,:] = s_im
    
            tr_lateral_lb = (s_lb % 9 )/3
            tr_head_lb    = (s_lb % 3)
            transferred_im_array[i*hparams.batch_size:(i+1)*hparams.batch_size,:,:,:] = tr_im
    
            tg_lateral_lb = (tg_lb % 9)/3
            tg_head_lb    = tg_lb % 3
            target_im_array[i*hparams.batch_size:(i+1)*hparams.batch_size,:,:,:] = tg_im
      
            if FLAGS.mode == 'marginal':
                source_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = 0 #s_lb
                transferred_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]       = 1 #s_lb + num_class
                target_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = 2 #tg_lb + 2 * num_class
            elif FLAGS.mode == 'lateral':
                #source_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = s_lateral_lb
                transferred_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]       = tr_lateral_lb 
                target_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = tg_lateral_lb + num_class
            elif FLAGS.mode == 'head':
                #source_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = s_head_lb
                transferred_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]       = tr_head_lb 
                target_lb_array[i*hparams.batch_size:(i+1)*hparams.batch_size]            = tg_head_lb + num_class
            else:
                raise NotImplementedError
    
        print('TSNE start') 
        if FLAGS.mode == 'marginal':
            im_concat = np.concatenate([source_im_array,transferred_im_array,target_im_array],0)
            lb_concat = np.concatenate([source_lb_array,transferred_lb_array,target_lb_array],0)
            total_im_num = 3 * FLAGS.filenum 
        else:
            im_concat = np.concatenate([transferred_im_array,target_im_array],0)
            lb_concat = np.concatenate([transferred_lb_array,target_lb_array],0)
            total_im_num = 2 * FLAGS.filenum

        # im_proj: embedded_data	
        if FLAGS.embed_dim == 2:
            im_proj2 = TSNE(n_components=2, random_state=FLAGS.RS, learning_rate=700., n_iter=240000).fit_transform(np.reshape(im_concat,(total_im_num,90*160*3)))
            #kPCA2 = KernelPCA(n_components=2, kernel='linear', gamma=15)
            #X_kpca2 = kPCA2.fit_transform(np.reshape(im_concat, (total_im_num, 90*160*3)))
            #iso = Isomap(n_neighbors=9, n_components=2)
            #iso.fit(np.reshape(im_concat, (total_im_num, 90*160*3)))
            #isomap2 = iso.transform(np.reshape(im_concat, (total_im_num, 90*160*3)))
            
            tsne_figure2, _,_,_= scatter2D(im_proj2, lb_concat, 'tsne')
            #kpca_figure2, _,_,_ = scatter2D(X_kpca2, lb_concat, 'kpca')
            #iso_figure2, _,_,_ = scatter2D(isomap2, lb_concat, 'iso')

        else:
            im_proj3 = TSNE(n_components=3, random_state=FLAGS.RS, learning_rate=700., n_iter=3000).fit_transform(np.reshape(im_concat,(total_im_num,90*160*3)))
            kPCA3 = KernelPCA(n_components=3, kernel='rbf', gamma=15)
            X_kpca3 = kPCA3.fit_transform(np.reshape(im_concat, (total_im_num, 90*160*3)))
            tsne_figure3, _, _ = scatter3D(im_proj3, lb_concat, 'tsne')
            kpca_figure3,_,_ = scatter3D(X_kpca3, lb_concat, 'kpca')

        
#        with open('tSNE_data.pickle','wb') as f:
#            pickle.dump([im_proj,lb_concat],f)
#            f.close()
#        with open('kPCA_data.pickle', 'wb') as f:
#            pickle.dump([X_kpca, lb_concat],f)
#            f.close()

        '''Tensorboard embedding'''
#        embedding_var = tf.Variable(im_concat, name='embedding_var')
#        sess.run(tf.global_variables_initializer())
#        tensorboard_saver = tf.train.Saver([embedding_var])
#        writer = tf.summary.FileWriter(log_dir + '/tsne', sess.graph)
#
#        config = projector.ProjectorConfig()
#        embeddings = config.embeddings.add()
#        embeddings.tensor_name = embedding_var.name     
#        # csv file to save labels
#        embeddings.metadata_path = os.path.join(log_dir + '/tsne', 'metadata.csv')
#        # sprite image path
#        embeddings.sprite.image_path = os.path.join(log_dir + '/tsne', 'sprite_image.png')
#        embeddings.sprite.single_image_dim.extend([28,28])
#        
#        projector.visualize_embeddings(writer, config)
#        tensorboard_saver.save(sess, os.path.join(log_dir + '/tsne', 'model.ckpt'))    
#
#        with open(embeddings.metadata_path, 'wb') as f:
#            f.write("Index\tLabel\n")
#            for index,labels in enumerate(lb_concat):
#                f.write('%d\t%d\n' %(index,labels))
#        
#        writer.close()
     
        coord.request_stop()
        coord.join(threads)
        print('TNSE finished')


def main():
  hparams = create_hparams(os.path.join(FLAGS.checkpoint_dir,'hparams.json'))
  log_dir = FLAGS.log_dir
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MkDir(log_dir)
  visualize(checkpoint_dir=FLAGS.checkpoint_dir,
            hparams=hparams,
            log_dir = log_dir)

if __name__ == "__main__":
  main()
