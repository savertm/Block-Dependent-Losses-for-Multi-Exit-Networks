#This code is for Improving Los-Latency Predictions in Multi-Exit Architectures via Block-Dependent Losses
#Reference
#The original backbone code comes from
#https://github.com/kalviny/MSDNet-PyTorch




import torch.nn as nn
import torch
import math
import pdb


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                  bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                  bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * args.grFactor[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        #pdb.set_trace()

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet_overlap(nn.Module):
    def __init__(self, args):
        super(MSDNet_overlap, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args

        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                              else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn,
                                                                                           args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(
                    1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x):
        res = []

        if self.args.method == 'overlap':

            if self.args.include_final == 'False':

                x1 = self.blocks[0](x)
                res.append(self.classifier[0](x1))

                x2 = self.blocks[1](x1)
                res.append(self.classifier[1](x2))
                x2_aux = self.blocks[1]([x_com.clone().detach() for x_com in x1])

                x3 = self.blocks[2](x2)
                res.append(self.classifier[2](x3))
                x3_aux = self.blocks[2](x2_aux)
                x3_aux_1 = self.blocks[2]([x_com.clone().detach() for x_com in x2_aux])

                x4 = self.blocks[3](x3_aux)
                res.append(self.classifier[3](x4))
                x4_aux = self.blocks[3](x3_aux_1)
                x4_aux_1 = self.blocks[3]([x_com.clone().detach() for x_com in x3_aux])

                x5 = self.blocks[4](x4_aux)
                res.append(self.classifier[4](x5))
                x5_aux = self.blocks[4](x4_aux_1)
                x5_aux_1 = self.blocks[4]([x_com.clone().detach() for x_com in x4_aux])

                x6 = self.blocks[5](x5_aux)
                res.append(self.classifier[5](x6))
                x6_aux = self.blocks[5](x5_aux_1)

                x7 = self.blocks[6](x6_aux)
                res.append(self.classifier[6](x7))


            else:

                x1 = self.blocks[0](x)
                res.append(self.classifier[0](x1))
                x1_final = self.blocks[0](x.clone().detach())

                x2 = self.blocks[1](x1)
                res.append(self.classifier[1](x2))
                x2_aux = self.blocks[1]([x_com.clone().detach() for x_com in x1])
                x2_final = self.blocks[1](x1_final)

                x3 = self.blocks[2](x2)
                res.append(self.classifier[2](x3))
                x3_aux = self.blocks[2](x2_aux)
                x3_aux_1 = self.blocks[2]([x_com.clone().detach() for x_com in x2_aux])
                x3_final = self.blocks[2](x2_final)

                x4 = self.blocks[3](x3_aux)
                res.append(self.classifier[3](x4))
                x4_aux = self.blocks[3](x3_aux_1)
                x4_aux_1 = self.blocks[3]([x_com.clone().detach() for x_com in x3_aux])
                x4_final = self.blocks[3](x3_final)

                x5 = self.blocks[4](x4_aux)
                res.append(self.classifier[4](x5))
                x5_aux = self.blocks[4](x4_aux_1)
                x5_final = self.blocks[4](x4_final)

                x6 = self.blocks[5](x5_aux)
                res.append(self.classifier[5](x6))
                x6_final = self.blocks[5](x5_final)

                x7 = self.blocks[6](x6_final)
                res.append(self.classifier[6](x7))



        elif self.args.method == 'local_error':
            x1 = self.blocks[0](x)
            res.append(self.classifier[0](x1))

            x2 = self.blocks[1]([x_com.clone().detach() for x_com in x1])
            res.append(self.classifier[1](x2))

            x3 = self.blocks[2]([x_com.clone().detach() for x_com in x2])
            res.append(self.classifier[2](x3))

            x4 = self.blocks[3]([x_com.clone().detach() for x_com in x3])
            res.append(self.classifier[3](x4))

            x5 = self.blocks[4]([x_com.clone().detach() for x_com in x4])
            res.append(self.classifier[4](x5))

            x6 = self.blocks[5]([x_com.clone().detach() for x_com in x5])
            res.append(self.classifier[5](x6))

            x7 = self.blocks[6]([x_com.clone().detach() for x_com in x6])
            res.append(self.classifier[6](x7))

        elif self.args.method == 'no_overlap':
            x1 = self.blocks[0](x)
            res.append(self.classifier[0](x1))

            x2 = self.blocks[1](x1)
            res.append(self.classifier[1](x2))

            x3 = self.blocks[2](x2)
            res.append(self.classifier[2](x3))

            x4 = self.blocks[3]([x_com.clone().detach() for x_com in x3])
            res.append(self.classifier[3](x4))

            x5 = self.blocks[4](x4)
            res.append(self.classifier[4](x5))

            x6 = self.blocks[5](x5)
            res.append(self.classifier[5](x6))

            x7 = self.blocks[6](x6)
            res.append(self.classifier[6](x7))


        return res

