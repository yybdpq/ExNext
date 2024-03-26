import torch
from torch import nn
from layer.embedding import CheckinEmbedding, EdgeEmbedding
from layer.transf import HyperghTransf
from layer.embedding import TimeEmbedding, DistanceEmbeddingHSTLSTM, DistanceEmbedding_STAN, DistanceEmbedding_S


class EXNEXT(nn.Module):
    def __init__(self, cfg):
        super(EXNEXT, self).__init__()
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        self.do_traj2traj = cfg.model_args.do_traj2traj
        self.distance_encoder_type = cfg.model_args.distance_encoder_type
        self.dropout_rate = cfg.model_args.dropout_rate
        self.generate_edge_attr = cfg.model_args.generate_edge_attr
        self.num_conv_layers = len(cfg.model_args.sizes)
        self.num_poi = cfg.dataset_args.num_poi
        self.embed_fusion_type = cfg.model_args.embed_fusion_type
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=cfg.dataset_args
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.checkin_embed_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )

        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        if cfg.conv_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embed_size
        else:
            continuous_encoder_dim = cfg.model_args.st_embed_size

        if self.generate_edge_attr:
            #use edge_type to create edge_attr_embed
            self.edge_attr_embedding_layer = EdgeEmbedding(
                embed_size=self.checkin_embed_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type
            )
        else:
            #source_traj_size, target_traj_size, jaccard similarity as the raw features, and do linear transformation
            if cfg.conv_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
            else:
                self.edge_attr_embedding_layer = None

        self.conv_list = nn.ModuleList()

        #conv for ci2traj within which some ci2traj relations have been removed by time to prevent data leakage
        self.conv_for_time_filter = HyperghTransf(
            in_channels=self.checkin_embed_size,
            out_channels=self.checkin_embed_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
            time_fusion_mode=cfg.conv_args.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False,
            adj_mask_matrix = None
        )
        self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
        self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

        if self.do_traj2traj:
            for i in range(self.num_conv_layers):
                if i == 0:
                    #ci2traj full
                    have_query_feature = False
                    residual_fusion_mode = None
                    edge_size = None
                else:
                    #traj2traj
                    have_query_feature = True
                    residual_fusion_mode = cfg.conv_args.residual_fusion_mode
                    if self.edge_attr_embedding_layer is None:
                        edge_size = 3
                    else:
                        edge_size = self.checkin_embed_size

                self.conv_list.append(
                    HyperghTransf(
                        in_channels=self.checkin_embed_size,
                        out_channels=self.checkin_embed_size,
                        attn_heads=cfg.conv_args.num_attention_heads,
                        residual_beta=cfg.conv_args.residual_beta,
                        learn_beta=cfg.conv_args.learn_beta,
                        dropout=cfg.conv_args.conv_dropout_rate,
                        trans_method=cfg.conv_args.trans_method,
                        edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
                        time_fusion_mode=cfg.conv_args.time_fusion_mode,
                        head_fusion_mode=cfg.conv_args.head_fusion_mode,
                        residual_fusion_mode=residual_fusion_mode,
                        edge_dim=edge_size,
                        rel_embed_dim=self.checkin_embed_size,
                        time_embed_dim=continuous_encoder_dim,
                        dist_embed_dim=continuous_encoder_dim,
                        negative_slope=cfg.conv_args.negative_slope,
                        have_query_feature=have_query_feature,
                        adj_mask_matrix = None
                    )
                )
            self.norms_list = nn.ModuleList()
            for i in range(self.num_conv_layers):
                self.norms_list.append(nn.BatchNorm1d(self.checkin_embed_size))

            self.dropout_list = nn.ModuleList()
            for i in range(self.num_conv_layers):
                self.dropout_list.append(nn.Dropout(self.dropout_rate))

        self.continuous_time_encoder = TimeEmbedding(cfg.model_args, continuous_encoder_dim)

        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEmbedding_STAN(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'time':
            self.continuous_distance_encoder = TimeEmbedding(cfg.model_args, continuous_encoder_dim)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEmbeddingHSTLSTM(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEmbedding_S(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        else:
            raise ValueError(f"Get wrong distance_encoder_type argument: {cfg.model_args.distance_encoder_type}!")

        self.linear = nn.Linear(self.checkin_embed_size, self.num_poi)
        
    def adj_mask(self, attention):
        mask_a_prob = torch.clamp(torch.sigmoid(attention), 0.001, 0.999)
        mask_a_matrix = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            torch.Tensor([0.05]).cuda(),
            probs=mask_a_prob).rsample()
        eps = 0.8
        mask_a_matrix = (mask_a_matrix > eps).detach().float()
        return mask_a_prob, mask_a_matrix
    
    def forward(self, data, label=None, adj_mask_matrix = None, mode='train'):
        input_x = data['x']
        split_idx = data['split_index']

        check_in_x = input_x[split_idx+1:]
        checkin_feature = self.checkin_embedding_layer(check_in_x)
        trajectory_feature = torch.zeros(
            split_idx+1,
            self.checkin_embed_size,
            device=checkin_feature.device
        )
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)

        edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='ch2tj')
        else:
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
            edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

        x_for_time_filter, attention_scores, attn_output_weights = self.conv_for_time_filter(  #[64,640]
            x,
            edge_index=data['edge_index'][0],
            edge_attr_embed=edge_attr_embed,
            edge_time_embed=edge_time_embed,
            edge_dist_embed=edge_distance_embed,
            edge_type_embed=edge_type_embed,
            mode = mode,
            adj_mask_matrix = adj_mask_matrix
        )
        x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

        if data['edge_index'][-1] is not None and self.do_traj2traj:    
            #all conv
            for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
                    zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:],
                        data["edge_type"][1:])
            ):
                edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='tj2tj')
                else:
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis)

                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    else:
                        if self.edge_attr_embedding_layer:
                            edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
                        else:
                            edge_attr_embed = edge_attr.to(torch.float32)

                if idx == len(data['edge_index']) - 2:
                    if mode in ('test', 'validate'):
                        batch_size = self.eval_batch_size
                    else:
                        batch_size = self.batch_size 
                    x_target = x_for_time_filter[:batch_size]   
                else:
                    x_target = x[:edge_index.sparse_sizes()[0]]

                x, attention_scores, attn_output_weights = self.conv_list[idx](    #attention_scores [n,4]
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed,
                    mode = mode,
                    adj_mask_matrix = adj_mask_matrix
                )
                x = self.norms_list[idx](x)
                x = self.act(x)
                x = self.dropout_list[idx](x)
        else:
            x = x_for_time_filter
        out = self.linear(x) 

        adj_mask_prob = None
        adj_mask_matrix = None

        if mode in ('train'):
            adj_mask_prob, adj_mask_matrix = self.adj_mask(attention_scores)   
        
        return out, adj_mask_prob, adj_mask_matrix
