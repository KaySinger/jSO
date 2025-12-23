# jSO
基于iLSHADE的改进算法。
J. Brest, M. S. Maučec and B. Bošković, "Single objective real-parameter optimization: Algorithm jSO," 2017 IEEE Congress on Evolutionary Computation (CEC), Donostia, Spain, 2017, pp. 1311-1318, doi: 10.1109/CEC.2017.7969456.

jSO基于iLSHADE算法进行改进得到，主要实现单目标优化问题的最值优化。考虑到iLSHADE和jSO的相似且源于同一作者，本人直接用iLSHADE的部分改进引入jSO的改进。
# iL-SHADE
作者主要在参数自适应机制进行了多项改进以提升L-SHADE的性能：

1、将成功记忆历史的初始参数设置为MF=0.5，MCR=0.8，更高的初始交叉率，并且使其中一个历史永远保存为MF=MCR=0.9，旨在在迭代后期算法仍然存在高变异高交叉的活力。

2、变异因子F采用三阶段参数限制：g<0.25Gmax，F=min(F,0.7)，g<0.5Gmax，F=min(F,0.8)，g<0.75Gmax，F=min(F,0.9)。
同样交叉因子CR采用二阶段参数限制：g<0.25Gmax，CR=max(CR,0.5)，g<0.5Gmax，CR=max(CR,0.25)。

3、参数的更新改为加权Lehmer均值，以F的更新为例：**MF ,k,g+1 ← (meanW L(SF ) + MF ,k,g )/2**

4、贪婪系数p随函数评估次数递增（0.1-0.2），确保先进个体的选择范围。

5、初始种群为12*D。
# jSO
jSO在iL-SHADE的基础上对细节进行修正以进一步提高iL-SHADE的算法性能。

1、将成功记忆历史的初始参数设置为MF=0.3，MCR=0.8，虽然作者在伪代码中MF设置为0.5，但是文段和代码中将MF均设置为0.3，因此本人仍将MF设置为0.3。

2、g<0.6Gmax，F=min(F,0.7)，交叉因子CR仍采用二阶段参数限制：g<0.25Gmax，CR=max(CR,0.5)，g<0.5Gmax，CR=max(CR,0.6)。

3、变异策略改为"DE/current-to-pbest-w/1"，这源于作者采取了Fw的三阶段选择：Fw={0.7*F,nfes<0.2*max_nfes;0.8*F,nfes<0.4*max_nfes;1.2*F,otherwise}，
因此变异策略变为vi=xi+Fw*(xpbest-xi)+F*(r1-r2)。

4、贪婪系数p随函数评估次数递增（0.125-0.25）。

5、初始种群为25log(D)√(D)。