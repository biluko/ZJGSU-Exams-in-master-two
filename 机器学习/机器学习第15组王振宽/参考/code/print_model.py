from torchstat import stat
from final_model import Model


model = Model()
stat(model, (1, 28, 28))
# model_1 = Model_1()
# stat(model_1, (1, 28, 28))
#
# model_2 = Model_2()
# stat(model_2, (1, 28, 28))
#
# model_3 = Model_3()
# stat(model_3, (1, 28, 28))
#
# model_4 = Model_4()
# stat(model_4, (1, 28, 28))
#
# model_5 = Model_5()
# stat(model_5, (1, 28, 28))

# print(model_1)
# print(model_2)
# print(model_3)
# print(model_4)