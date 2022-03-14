#include <exception>
#include <iostream>
#include <string>
#include <memory>

#include "kart/dataset3.hpp"

using namespace std;
using namespace kart;

Dataset3::Dataset3(KartRepo *repo, Tree tree_)
    : repo(repo), tree_(tree_)
{
}

unique_ptr<Tree> Dataset3::get_tree()
{
    return make_unique<Tree>(tree_);
}
unique_ptr<Tree> Dataset3::get_feature_tree()
{
    auto entry = tree_.get_entry_by_path(DATASET_DIRNAME + "/feature");
    auto feature_tree = entry.get_object().as_tree();
    return make_unique<Tree>(feature_tree);
}

unique_ptr<BlobWalker> Dataset3::feature_blobs()
{
    auto feature_tree = get_feature_tree();
    return make_unique<BlobWalker>(repo, move(feature_tree));
}
