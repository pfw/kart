import pygit2

from .exceptions import NO_USER, NotFound


def peel_to_commit_and_tree(commitish_or_treeish):
    """
    Given a commitish or treeish object, returns a tuple of (commit, tree).
    If the given object is commitish, both commit and tree will be populated.
    If the given object is treeish, then commit will be None.
    """
    obj = commitish_or_treeish

    # These quick checks to see what .type is make it easy to implement commit-ish or tree-ish objects
    # without implementing peel.
    if obj.type == pygit2.GIT_OBJ_COMMIT and hasattr(obj, "tree"):
        return obj, obj.tree
    elif obj.type == pygit2.GIT_OBJ_TREE:
        return None, obj

    commit = _safe_peel(obj, pygit2.Commit)
    if commit is not None:
        return commit, commit.tree
    tree = _safe_peel(obj, pygit2.Tree)
    if tree is not None:
        return None, tree
    raise ValueError(f"Can't peel {obj!r} - to a commit or a tree")


def _safe_peel(obj, target_type):
    try:
        return obj.peel(target_type)
    except (pygit2.InvalidSpecError, AttributeError):
        return None


def all_blobs_in_tree(tree):
    """Recursively yields all possible blobs in the given directory tree."""
    for entry in tree:
        if entry.type == pygit2.GIT_OBJ_BLOB:
            yield entry
        elif entry.type == pygit2.GIT_OBJ_TREE:
            yield from all_blobs_in_tree(entry)


def all_trees_in_tree(tree, ignore_hidden=True):
    """
    Recursively yields all possible trees in the given directory tree.
    ignore_hidden - don't include trees which have a "." prefixed to their name.
    """
    for entry in tree:
        if entry.type != pygit2.GIT_OBJ_TREE:
            continue
        if ignore_hidden and entry.name.startswith("."):
            continue
        yield entry
        yield from all_trees_in_tree(entry)


def all_blobs_with_paths_in_tree(tree, path=""):
    """Recursively yields all possible (path, blob) tuples in the given directory tree."""
    for entry in tree:
        entry_path = f"{path}/{entry.name}" if path else entry.name
        if entry.type == pygit2.GIT_OBJ_BLOB:
            yield entry_path, entry
        elif entry.type == pygit2.GIT_OBJ_TREE:
            yield from all_blobs_with_paths_in_tree(entry, path=entry_path)


def all_trees_with_paths_in_tree(tree, path="", ignore_hidden=True):
    """
    Recursively yields all possible (path, tree) tuples in the given directory tree.
    ignore_hidden - don't include trees which have a "." prefixed to their name.
    """
    for entry in tree:
        if entry.type != pygit2.GIT_OBJ_TREE:
            continue
        if ignore_hidden and entry.name.startswith("."):
            continue
        entry_path = f"{path}/{entry.name}" if path else entry.name
        yield entry_path, entry
        yield from all_trees_with_paths_in_tree(entry, entry_path)


def walk_tree(top, path="", topdown=True):
    """
    Corollary of os.walk() for git Tree objects:

    For each subtree in the tree rooted at top (including top itself),
    yields a 4-tuple:
        top_tree, top_path, subtree_names, blob_names

    top_tree is a Tree object
    top_path is a string, the path to top_tree with respect to the root path.
    subtree_names is a list of names for the subtrees in top_tree
    blob_names is a list of names for the blobs in top_tree.

    To get a full path (which begins with top_path) to a blob or subtree in
    top_path, do `"/".join([top_path, name])`.

    To get a TreeEntry-style Blob or Tree object, do `top_tree / name`

    If optional arg `topdown` is true or not specified, the tuple for a
    subtree is generated before the tuples for any of its subtrees
    (pre-order traversal).  If topdown is false, the tuple
    for a subtree is generated after the tuples for all of its
    subtrees (post-order traversal).

    When topdown is true, the caller can modify the subtree_names list in-place
    (e.g., via del or slice assignment), and walk will only recurse into the
    subtrees whose names remain; this can be used to prune the
    search, or to impose a specific order of visiting.  Modifying subtree_names when
    topdown is false is ineffective, since the directories in subtree_names have
    already been generated by the time subtree_names itself is generated. No matter
    the value of topdown, the list of subtrees is retrieved before the
    tuples for the tree and its subtrees are generated.
    """
    subtree_names = []
    blob_names = []

    for entry in top:
        is_tree = entry.type == pygit2.GIT_OBJ_TREE

        if is_tree:
            subtree_names.append(entry.name)
        elif entry.type == pygit2.GIT_OBJ_BLOB:
            blob_names.append(entry.name)
        else:
            pass

    if topdown:
        yield top, path, subtree_names, blob_names
        for name in subtree_names:
            subtree_path = "/".join([path, name]) if path else name
            subtree = top / name
            yield from walk_tree(subtree, subtree_path, topdown=topdown)
    else:
        for name in subtree_names:
            subtree_path = "/".join([path, name]) if path else name
            subtree = top / name
            yield from walk_tree(subtree, subtree_path, topdown=topdown)
        yield top, path, subtree_names, blob_names


def check_git_user(repo=None):
    """
    Checks whether a user is defined in either the repo configuration or globally

    If not, errors with a semi-helpful message
    """
    if repo:
        cfg = repo.config
    else:
        try:
            cfg = pygit2.Config.get_global_config()
        except IOError:
            # there is no global config
            cfg = {}

    try:
        user_email = cfg["user.email"]
        user_name = cfg["user.name"]
        if user_email and user_name:
            return (user_email, user_name)
    except KeyError:
        pass

    msg = [
        "Please tell me who you are.",
        "\nRun",
        '\n  git config --global user.email "you@example.com"',
        '  git config --global user.name "Your Name"',
        "\nto set your account's default identity.",
    ]
    if repo:
        msg.append("Omit --global to set the identity only in this repository.")

    msg.append("\n(Kart uses the same credentials and configuration as git)")

    raise NotFound("\n".join(msg), exit_code=NO_USER)
