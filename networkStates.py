import math
import random
import numpy as np
# taken from github : METAQNN , with some modification:
# I added the possibility of parallel action that ended with a global average pooling in that cas
class Statearchi:
    def __init__(self,
                 layer_type=None,  # String -- conv, pool, fc, softmax,lstm
                 layer_depth=None,
                 terminate=None,# Current depth of network
                 batch_size=None,
                 nb_filter=None,  # Used for conv, 0 when not conv
                 filter_size=None,  # Used for conv and pool, 0 otherwise
                 stride=None,  # Used for conv and pool, 0 otherwise
                 image_size=None,  # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 fc_size=None,
                 lstm_size=None,
                 seq_lengh=None,
                 output_size=None,
                 complex=None,
                 nb_all_parallel=None,
                 nb_consecutif_conv=None,
                 nb_consecutif_pool=None,
                 nb_consecutif_fc=None,# Used for fc and softmax -- number of neurons in layer
                 state_list=None):  # can be constructed from a list instead, list takes precedent


        if not state_list:
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.batch_size = batch_size
            self.nb_filter = nb_filter  # Used for conv, 0 when not conv
            self.filter_size = filter_size  # Used for conv and pool, 0 otherwise
            self.stride = stride  # Used for conv and pool, 0 otherwise
            self.image_size = image_size  # Used for any layer that maintains square input (conv and pool), 0 otherwise
            self.fc_size = fc_size
            self.lstm_size = lstm_size  # lstm neuron_dim
            self.seq_lengh = seq_lengh
            self.complex=complex
            self.nb_all_parallel = nb_all_parallel
            self.output_size=output_size
            self.terminate=terminate
            self.nb_consecutif_conv=nb_consecutif_conv
            self.nb_consecutif_pool=nb_consecutif_pool
            self.nb_consecutif_fc=nb_consecutif_fc
        else:
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.batch_size = state_list[2]
            self.nb_filter = state_list[3]
            self.filter_size = state_list[4]
            self.stride = state_list[5]
            self.image_size = state_list[6]
            self.fc_size = state_list[7]
            self.lstm_size = state_list[8]
            self.output_size=state_list[9]
            self.seq_lengh = state_list[10]
            self.complex=state_list[11]
            self.nb_all_parallel = state_list[12]
            self.nb_consecutif_conv = state_list[13]
            self.nb_consecutif_pool = state_list[14]
            self.nb_consecutif_fc = state_list[15]

    def as_tuple(self):
            return (self.layer_type,
                    self.layer_depth,
                    self.filter_depth,
                    self.filter_size,
                    self.stride,
                    self.image_size,
                    self.fc_size,
                    self.seq_lengh,
                    self.batch_size,
                    self.seq_lengh,
                    self.lstm_size,
                    self.terminate,
                    self.nb_consecutif_conv,
                    self.nb_consecutif_pool,
                    self.nb_consecutif_fc,
                    self.complex,
                    self.nb_all_parallel
            )
class Statecomplex:
    def __init__(self,
               liststate_branch0,
               liststate_branch1,
               liststate_branch2,

               terminate
               ):
        self.liststate_branch0=liststate_branch0
        self.liststate_branch1=liststate_branch1
        self.liststate_branch2=liststate_branch2
        self.terminate=terminate

    def as_tupleC(self):
        return (self.liststate_branch0,
                self.liststate_branch1,
                self.liststate_branch1,
                self.terminate
                )

class enumerate:
    def __init__(self, state_space_parameters , layer_type):
        # Limits
        self.ssp = state_space_parameters
        self.layer_limit = state_space_parameters.layer_limi
        self.output_states = state_space_parameters.output_states
        self.layer_type=layer_type
    '''
    begin conv or lstm
    conv --> conv,pool,fc :hidden
    fc --> fc,lstm,conv: hidden
    pool--> pool,fc,conv:hidden
    always closed by fc+softmax terminal state
    max-consecutive conv=4,
    max-consecutive lstm=10,
    max consecutive fc=3,
    max consecutive pool=2,
    possibility that output of one layer is the input of multile layer --- must be integrated (not yet)
    in that case concat then global average pooling
    '''

    def enumerate_complex_actions(self,statearchi, q_values,count_parallel):
        #{branch= 0 or 1 or 2, actions=[[],[],[]],utilities=[[],[],[]]}
        import numpy as np
        list_all=[]
        for j in range(count_parallel-1):

            possible_action_list=[]
            possible_utility_action=[]
            dictenumer=self.enumerate_state(self, statearchi, q_values,complex=True)
            list_all.append(dictenumer)
        actioncomplex=[]
        for terminate_complex in [True,False]:
            if count_parallel==2:
                for action in list_all[0]['actions']:
                    for action2 in list_all[1]['actions']:
                        actioncomplex += Statecomplex(liststate_branch0=action, liststate_branch1=action2, liststate_branch2=None,
                                                      layer_deep=None, terminate=terminate_complex)

            if count_parallel == 3:
                for action in list_all[0]['actions']:
                    for action2 in list_all[1]['actions']:
                        for action3 in list_all[2]['actions']:
                            actioncomplex += Statecomplex(liststate_branch0=action, liststate_branch1=action2,
                                                          liststate_branch2=action3,
                                                          layer_deep=None, terminate=terminate_complex)




        return actioncomplex
    def transitional_state(self,statecomplex,q_values):
        action=[]
        action += Statearchi(layer_type='concat',
                             layer_depth=statecomplex.layer_depth + 1,
                             nb_filter=0,  # Used for conv, 0 when not conv
                             filter_size=0,
                             # Used for conv and pool, 0 otherwise
                             stride=1,  # Used for conv and pool, 0 otherwise
                             image_size=self._calc_new_image_concat(
                                 statecomplex,
                                 type='concat'),
                             # Used for any layer that maintains square input (conv and pool), 0 otherwise
                             fc_size=0,
                             lstm_size=0,  # lstm neuron_dim
                             seq_lengh=0,
                             terminate=0,
                             complex=False,
                             nb_consecutif_conv=0
                             )
        return action


    def enumerate_state(self, statearchi,statecomplex, q_values,complex=False):
        action = []
        complex_actions=[]
        for complex in [True,False]:
            if statecomplex!=None and complex==True:
                for i in [0,1,2]:
                    if i==0 and statecomplex.liststate_branch0!=None:
                        sizei=len(statecomplex.liststate_branch0)
                        list_all = self.enumerate_complex_actions(self, statecomplex.liststate_branch0[sizei-1], q_values, count_parallel=3)
                    else:
                        if i == 1 and statecomplex.liststate_branch1!=None:
                            sizei = len(statecomplex.liststate_branch1)
                            list_all = self.enumerate_complex_actions(self, statecomplex.liststate_branch1[sizei - 1],
                                                                      q_values, count_parallel=3)
                        else:
                            if i == 2 and statecomplex.liststate_branch2!=None:
                                sizei = len(statecomplex.liststate_branch2)
                                list_all = self.enumerate_complex_actions(self,
                                                                          statecomplex.liststate_branch2[sizei - 1],
                                                                          q_values, count_parallel=3)
                            else:
                                raise("exepetion in last state")
                complex_actions+=list_all

            else:
                if statecomplex!=None and complex==False:
                    action+=self.transitional_state(self,statecomplex,q_values)
                else:
                    if statearchi!=None and complex==True:
                        list_all = self.enumerate_complex_actions(self, statearchi, q_values, count_parallel=3)
                        complex_actions=+list_all
                    else:
                        if statearchi!=None and complex==False:

                            if statearchi.terminate == 0:

                                if statearchi.layer_depth < (self.layer_limit - 1):
                                    if statearchi.layer_type in ['conv', 'fc', 'pool']:
                                        if statearchi.layer_type == 'conv' and statearchi.nb_consecutif_conv < 3 \
                                                or statearchi.layer_type in ['fc', 'pool']:
                                            if statearchi.layer_depth == 0:

                                                for depth in self.ssp.possible_conv_depths:
                                                    for filt in self._possible_conv_sizes(statearchi.image_size):
                                                        action += Statearchi(layer_type='conv',
                                                                             layer_depth=statearchi.layer_depth + 1,
                                                                             nb_filter=depth,
                                                                             # Used for conv, 0 when not conv
                                                                             filter_size=filt,
                                                                             # Used for conv and pool, 0 otherwise
                                                                             stride=1,
                                                                             # Used for conv and pool, 0 otherwise
                                                                             image_size=self._calc_new_image_size(
                                                                                 statearchi.image_size, filt,
                                                                                 depth, type='conv'),
                                                                             # Used for any layer that maintains square input (conv and pool), 0 otherwise
                                                                             fc_size=0,
                                                                             lstm_size=0,  # lstm neuron_dim
                                                                             seq_lengh=0,
                                                                             terminate=0,
                                                                             complex=complex,
                                                                             nb_consecutif_conv=[(
                                                                                                     statearchi.nb_consecutif_conv + 1) if statearchi.layer_type == 'conv' else 1])  # à ajouter parallèle
                                            else:
                                                for depth in self.ssp.possible_conv_depths:

                                                    for filt in self._possible_conv_sizes(statearchi.image_size):
                                                        # if i==1 all consecutive action with parallel ==1 should be parallel
                                                        action += Statearchi(layer_type='conv',
                                                                             layer_depth=statearchi.layer_depth + 1,
                                                                             nb_filter=depth,
                                                                             # Used for conv, 0 when not conv
                                                                             filter_size=filt,
                                                                             # Used for conv and pool, 0 otherwise
                                                                             stride=1,
                                                                             # Used for conv and pool, 0 otherwise
                                                                             image_size=self._calc_new_image_size(
                                                                                 statearchi.image_size,
                                                                                 filt, depth,
                                                                                 type='conv'),
                                                                             fc_size=0,
                                                                             lstm_size=0,  # lstm neuron_dim
                                                                             seq_lengh=0,
                                                                             terminate=0,
                                                                             complex=complex,
                                                                             nb_all_parallel=statearchi.nb_all_parallel + 1,
                                                                             nb_consecutif_conv=[(
                                                                                                     statearchi.nb_consecutif_conv + 1) if statearchi.layer_type == 'conv' else 1])

                                    if statearchi.layer_type in ['conv', 'pool']:
                                        if statearchi.layer_type == 'pool' and statearchi.nb_consecutif_pool < 2 \
                                                or statearchi.layer_type == 'conv':

                                            for filt in self._possible_pool_sizes(statearchi.image_size):
                                                action += Statearchi(layer_type='pool',
                                                                     layer_depth=statearchi.layer_depth + 1,
                                                                     nb_filter=statearchi.nb_filter,
                                                                     # Used for conv, 0 when not conv
                                                                     filter_size=filt,
                                                                     # Used for conv and pool, 0 otherwise
                                                                     stride=1,  # Used for conv and pool, 0 otherwise
                                                                     image_size=self._calc_new_image_size(
                                                                         Statearchi.image_size, filt,
                                                                         type='pool'),
                                                                     # Used for any layer that maintains square input (conv and pool), 0 otherwise
                                                                     fc_size=0,
                                                                     lstm_size=0,  # lstm neuron_dim
                                                                     seq_lengh=0,
                                                                     parallel=complex,
                                                                     nb_consecutif_pool=[
                                                                         statearchi.nb_consecutif_pool + 1 if statearchi.layer_type == 'pool' else 1],
                                                                     terminate=0)

                                    if statearchi.layer_type in ['lstm', 'fc']:
                                        if statearchi.layer_type == 'lstm' and self.ssp.allowconsecutive_lstm \
                                                or statearchi.layer_type in ['fc']:
                                            for seq in self._possible_seq_len(statearchi.image_size):
                                                action += Statearchi(layer_type='lstm',
                                                                     layer_depth=statearchi.layer_depth + 1,
                                                                     nb_filter=0,  # Used for conv, 0 when not conv
                                                                     filter_size=0,
                                                                     # Used for conv and pool, 0 otherwise
                                                                     stride=0,  # Used for conv and pool, 0 otherwise
                                                                     image_size=self._calc_new_image_size(
                                                                         Statearchi.image_size, seq,
                                                                         type='lstm'),
                                                                     # Used for any layer that maintains square input (conv and pool), 0 otherwise
                                                                     fc_size=0,
                                                                     lstm_size=
                                                                     self._calc_new_lstm(seq, self._calc_new_image_size(
                                                                         Statearchi.image_size, filt))[2],
                                                                     # lstm neuron_dim
                                                                     seq_lengh=seq,
                                                                     nb_all_parallel=0,
                                                                     complex=complex,
                                                                     terminate=0)

                                    if statearchi.layer_type in ['fc', 'pool', 'conv', 'lstm']:
                                        if statearchi.layer_type == 'fc' and statearchi.nb_consecutif_fc < 2 \
                                                or statearchi.layer_type in ['pool', 'conv', 'lstm']:
                                            for seq in self._possible_fc_size(statearchi.image_size):
                                                action += Statearchi(layer_type='fc',
                                                                     layer_depth=statearchi.layer_depth + 1,
                                                                     nb_filter=0,  # Used for conv, 0 when not conv
                                                                     filter_size=0,
                                                                     # Used for conv and pool, 0 otherwise
                                                                     stride=0,  # Used for conv and pool, 0 otherwise
                                                                     image_size=self._calc_new_image_size(
                                                                         Statearchi.image_size, type='fc'),
                                                                     # Used for any layer that maintains square input (conv and pool), 0 otherwise
                                                                     fc_size=seq,
                                                                     lstm_size=0,  # lstm neuron_dim
                                                                     seq_lengh=0,
                                                                     nb_consecutif_fc=[(
                                                                                           statearchi.nb_consecutif_fc + 1) if statearchi.layer_type == 'fc' else 1],
                                                                     complex=complex,
                                                                     terminate=0)

                                else:
                                    if statearchi.layer_depth == (self.layer_limit - 1):
                                        action += Statearchi(layer_type='fc',
                                                             nb_filter=0,
                                                             filter_size=0,
                                                             stride=0,
                                                             image_size=self._calc_new_image_size(Statearchi.image_size,
                                                                                                  type='fc'),
                                                             layer_depth=statearchi.layer_depth + 1,
                                                             fc_size=statearchi.output_size,
                                                             complex=False,
                                                             lstm_size=0,
                                                             seq_lengh=0,
                                                             terminate=1)
                                        action += Statearchi(layer_type='softmax',
                                                             complex=False,
                                                             layer_depth=statearchi.layer_depth,
                                                             terminate=1)





        q_values[statearchi.as_tuple()]= {'actions': [(to_state.as_tuple()) for to_state in action],'utilitiesS': [0.5 for i in range(len(action))],
                                          'actioncomplex':[(to_stateC.as_tupleC())for to_stateC in complex_actions],
                                          'utilitiesC': [0.5 for i in range(len(complex_actions))]  }


        return q_values



    def _calc_new_image_size(image_size,nb_filter,type,size_filter,seq_len):
        new_size=[]
        if type=='conv':
            new_size.append(nb_filter)
            new_size.append(int(math.ceil(float(image_size[0] - size_filter[0] + 1) / float(1))))
            new_size.append(int(math.ceil(float(image_size[1] - size_filter[1] + 1) / float(1))))
        if type=='lstm':
            x=1
            for i in image_size:
                x=x*i
            if x % seq_len == 0:
                new_size.append(seq_len)
                new_size.append(int(math.ceil(x/seq_len)))
        return new_size

    def _calc_new_image_concat(self, statecomplex):
        new_size=[]
        x=statecomplex.liststate_branch0
        y=statecomplex.liststate_branch1
        z=statecomplex.liststate_branch2
        if x!=None and y!=None and z!=None:
            sizex=x.image_size
            lx, wx = sizex[0], sizex[1]

            sizey=y.image_size
            ly, wy = sizey[0], sizey[1]

            sizez=z.image_size
            lz, wz = sizez[0], sizez[1]

            if lx==ly==lz:#concat horizontal
                new_size.append(lx)
                new_size.append(wx+wy+wz)
            else:
                if wx==wy==wz:#concat vertical
                    new_size.append(lx + ly + lz)
                    new_size.append(wx)
                else:
                    raise "image size of different branches are not compatible for concatenation"
        else:
            if x!=None and y!=None:
                sizex = x.image_size
                lx, wx = sizex[0], sizex[1]

                sizey = y.image_size
                ly, wy = sizey[0], sizey[1]
                if lx == ly :  # concat horizontal
                    new_size.append(lx)
                    new_size.append(wx + wy )
                else:
                    if wx == wy:  # concat vertical
                        new_size.append(lx + ly )
                        new_size.append(wx)
                    else:
                        raise "image size of different branches are not compatible for concatenation"
            else:
                if y != None and z != None:
                    sizey = y.image_size
                    ly, wy = sizey[0], sizey[1]

                    sizez = z.image_size
                    lz, wz = sizez[0], sizez[1]
                    if lz == ly:  # concat horizontal
                        new_size.append(lz)
                        new_size.append(wy + wz)
                    else:
                        if wy == wz:  # concat vertical
                            new_size.append(ly + lz)
                            new_size.append(wz)
                        else:
                            raise "image size of different branches are not compatible for concatenation"
                else:
                    if x != None and z != None:
                        sizex = x.image_size
                        lx, wx = sizex[0], sizex[1]

                        sizez = z.image_size
                        lz, wz = sizez[0], sizez[1]
                        if lz == lx:  # concat horizontal
                            new_size.append(lx)
                            new_size.append(wx + wz)
                        else:
                            if wx == wz:  # concat vertical
                                new_size.append(lx + lz)
                                new_size.append(wz)
                            else:
                                raise "image size of different branches are not compatible for concatenation"
        return new_size








    def _possible_conv_sizes(self, image_size):
        return [conv for conv in self.ssp.possible_conv_sizes if conv < image_size]

    def _possible_pool_sizes(self, image_size):
        return [pool for pool in self.ssp.possible_pool_sizes if pool < image_size]

    def _possible_seq_len(self, image_size):
        x = 1
        for i in image_size:
            x = x * i
        return [seq for seq in self.ssp._possible_seq_len if x%seq==0]


    def _possible_fc_size(self, state):
        '''Return a list of possible FC sizes given the current state'''
        if state.layer_type == 'fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes





