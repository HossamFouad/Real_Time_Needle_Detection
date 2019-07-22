import tensorflow as tf


def calculateCost(estimatedHeatmaps, gtHeatmaps, numberOfHeatmaps=3):
    cost = 0
    individualCosts = []  # for logging stageSpecific costs
    heatMapCosts = []  # for logging heatmapSpecific costs
    for element in estimatedHeatmaps:
        individualCost = tf.nn.l2_loss((element - gtHeatmaps))
        cost = cost + individualCost
        individualCosts.append(individualCost)
    for i in range(0, numberOfHeatmaps):
        heatMapCosts.append(0)
        for element in estimatedHeatmaps:
            heatMapCosts[i] = heatMapCosts[i] + tf.nn.l2_loss(element[:, :, :, i] - gtHeatmaps[:, :, :, i])

    return cost, individualCosts, heatMapCosts

