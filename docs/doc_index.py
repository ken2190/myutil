

utilmy/graph.py


utilmy/decorators.py
-------------------------functions----------------------
thread_decorator(func)
timeout_decorator(seconds = 10, error_message = os.strerror(errno.ETIME)
timer_decorator(func)
profiler_context()
profiler_decorator(func)
profiler_decorator_base(fnc)
test0()
thread_decorator_test()
profiler_decorator_base_test()
timeout_decorator_test()
profiled_sum()
dummy_func()



utilmy/adatasets.py
-------------------------functions----------------------
test0()
test1()
log(*s)
log2(*s)
dataset_classifier_XXXXX(nrows = 500, **kw)
pd_train_test_split(df, coly = None)
pd_train_test_split2(df, coly)
dataset_classifier_pmlb(name = '', return_X_y = False)
test_dataset_classifier_covtype(nrows = 500)
test_dataset_regression_fake(nrows = 500, n_features = 17)
test_dataset_classification_fake(nrows = 500)
test_dataset_classification_petfinder(nrows = 1000)
fetch_dataset(url_dataset, path_target = None, file_target = None)



utilmy/utilmy.py
-------------------------functions----------------------
import_function(fun_name = None, module_name = None)
help_create(modulename = 'utilmy.nnumpy', prefixs = None)
pd_random(ncols = 7, nrows = 100)
pd_generate_data(ncols = 7, nrows = 100)
pd_getdata(verbose = True)
git_repo_root()
git_current_hash(mode = 'full')
save(dd, to_file = "", verbose = False)
load(to_file = "")

-------------------------methods----------------------
Session.__init__(self, dir_session = "ztmp/session/", )
Session.show(self)
Session.save(self, name, glob = None, tag = "")
Session.load(self, name, glob:dict = None, tag = "")
Session.save_session(self, folder, globs, tag = "")
Session.load_session(self, folder, globs = None)


utilmy/debug.py
-------------------------functions----------------------
log(*s)
help()
print_everywhere()
log10(*s, nmax = 60)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
profiler_start()
profiler_stop()



utilmy/nnumpy.py


utilmy/__init__.py


utilmy/oos.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
test0()
test1()
test2()
test4()
test5()
to_dict(**kw)
to_timeunix(datex = "2018-01-16")
to_datetime(x)
np_list_intersection(l1, l2)
np_add_remove(set_, to_remove, to_add)
to_float(x)
to_int(x)
is_int(x)
is_float(x)
os_path_size(path  =  '.')
os_path_split(fpath:str = "")
os_file_replacestring(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_walk(path, pattern = "*", dirlevel = 50)
os_copy_safe(dirin = None, dirout = None, nlevel = 10, nfile = 100000, cmd_fallback = "")
z_os_search_fast(fname, texts = None, mode = "regex/str")
os_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_get_function_name()
os_variable_init(ll, globs)
os_import(mod_name = "myfile.config.model", globs = None, verbose = True)
os_variable_exist(x, globs, msg = "")
os_variable_check(ll, globs = None, do_terminate = True)
os_clean_memory(varlist, globx)
os_system_list(ll, logfile = None, sleep_sec = 10)
os_file_check(fp)
os_to_file(txt = "", filename = "ztmp.txt", mode = 'a')
os_platform_os()
os_cpu()
os_platform_ip()
os_memory()
os_sleep_cpu(cpu_min = 30, sleep = 10, interval = 5, msg =  "", verbose = True)
os_sizeof(o, ids, hint = " deep_getsizeof(df_pd, set()
os_copy(dirin = None, dirout = None, nlevel = 10, nfile = 100000, cmd_fallback = "")
os_removedirs(path)
os_getcwd()
os_system(ll, logfile = None, sleep_sec = 10)
os_makedirs(dir_or_file)
print_everywhere()
log10(*s, nmax = 60)
log5(*s)
log_trace(msg = "", dump_path = "", globs = None)
profiler_start()
profiler_stop()

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


utilmy/distributed.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
log_mem(*s)
test1_functions()
test2_funtions_thread()
test3_index()
test_all()
os_lock_acquireLock(plock:str = "tmp/plock.lock")
os_lock_releaseLock(locked_file_descriptor)
os_lock_execute(fun_run, fun_args = None, ntry = 5, plock = "tmp/plock.lock")
date_now(fmt = "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S %Z%z")
time_sleep_random(nmax = 5)
save(dd, to_file = "", verbose = False)
load(to_file = "")
load_serialize(name)
save_serialize(name, value)

-------------------------methods----------------------
IndexLock.__init__(self, findex, plock)
IndexLock.get(self)
IndexLock.put(self, val = "", ntry = 100, plock = "tmp/plock.lock")


utilmy/dates.py
-------------------------functions----------------------
test()
random_dates(start, end, size)
random_genders(size, p = None)
log(*s)
pd_date_split(df, coldate  =   'time_key', prefix_col  = "", verbose = False)
date_now(fmt="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S %Z%z", add_days = 0, timezone = 'Asia/Tokyo')
date_is_holiday(array)
date_weekmonth2(d)
date_weekmonth(d)
date_weekyear2(dt)
date_weekday_excel(x)
date_weekyear_excel(x)
date_generate(start = '2018-01-01', ndays = 100)



utilmy/ppandas.py
-------------------------functions----------------------
pd_dtype_count_unique(df, col_continuous = [])
pd_dtype_to_category(df, col_exclude, treshold = 0.5)
pd_dtype_getcontinuous(df, cols_exclude:list = [], nsample = -1)
pd_del(df, cols:list)
pd_add_noise(df, level = 0.05, cols_exclude:list = [])
pd_cols_unique_count(df, cols_exclude:list = [], nsample = -1)
pd_show(df, nrows = 100, reader = 'notepad.exe', **kw)
to_dict(**kw)
to_timeunix(datex = "2018-01-16")
to_datetime(x)
np_list_intersection(l1, l2)
np_add_remove(set_, to_remove, to_add)
to_float(x)
to_int(x)
is_int(x)
is_float(x)

-------------------------methods----------------------
dict_to_namespace.__init__(self, d)


utilmy/keyvalue.py
-------------------------functions----------------------
os_environ_set(name, value)
os_path_size(folder = None)
db_init(db_dir:str = "path", globs = None)
db_flush(db_dir)
db_size(db_dir =  None)
db_merge()
db_create_dict_pandas(df = None, cols = None, colsu = None)
db_load_dict(df, colkey, colval, verbose = True)
diskcache_load(db_path_or_object = "", size_limit = 100000000000, verbose = True)
diskcache_save(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, tbreak = 1, ## Break during insert to prevent big WAL file**kw)
diskcache_save2(df, colkey, colvalue, db_path = "./dbcache.db", size_limit = 100000000000, timeout = 10, shards = 1, npool = 10, sqlmode =  'fast', verbose = True)
diskcache_getkeys(cache)
diskcache_keycount(cache)
diskcache_getall(cache, limit = 1000000000)
diskcache_get(cache)
diskcache_config(db_path = None, task = 'commit')

-------------------------methods----------------------
DBlist.__init__(self, config_dict = None, config_path = None)
DBlist.add(self, db_path)
DBlist.remove(self, db_path)
DBlist.list(self, show = True)
DBlist.info(self, )
DBlist.clean(self, )
DBlist.check(self, db_path = None)
DBlist.show(self, db_path = None, n = 4)


utilmy/parallel.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
pd_random(nrows = 1000, ncols =  5)
test_fun_sum_inv(group, name = None)
test_fun_sum(group, name = None)
test_fun_sum2(list_vars, const = 1, const2 = 1)
test_fun_run(list_vars, const = 1, const2 = 1)
test_run_multithread(thread_name, num, string)
test_run_multithread2(thread_name, arg)
test_sum(x)
test0()
test_pdreadfile()
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, nfile = 1000000, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, **kw)
pd_read_file2(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, nfile = 1000000, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, **kw)
pd_groupby_parallel2(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_groupby_parallel(df, colsgroup = None, fun_apply = None, npool: int  =  1, **kw, )
pd_apply_parallel(df, fun_apply = None, npool = 5, verbose = True)
multiproc_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multithread_run(fun_async, input_list: list, n_pool = 5, start_delay = 0.1, verbose = True, input_fixed:dict = None, npool = None, **kw)
multiproc_tochunk(flist, npool = 2)
multithread_run_list(**kwargs)
z_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, **kw)
zz_pd_read_file3(path_glob = "*.pkl", ignore_index = True, cols = None, nrows = -1, concat_sort = True, n_pool = 1, npool = None, drop_duplicates = None, col_filter = None, col_filter_val = None, dtype_reduce = None, fun_apply = None, max_file = -1, #### apply function for each subverbose = False, **kw)
zz_pd_groupby_parallel5(df, colsgroup = None, fun_apply = None, npool = 5, verbose = False, **kw)
ztest1()
ztest2()



utilmy/tabular.py
-------------------------functions----------------------
test0()
test1()
test3()
log(*s)
y_adjustment()
test_anova(df, col1, col2)
test_normality2(df, column, test_type)
test_plot_qqplot(df, col_name)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_normality(df, column, test_type)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_hypothesis(df_obs, df_ref, method = '', **kw)
estimator_std_normal(err, alpha = 0.05, )
estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
pd_train_test_split_time(df, test_period  =  40, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)
pd_to_scipy_sparse_matrix(df)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
np_col_extractname(col_onehot)
np_list_remove(cols, colsremove, mode = "exact")
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")



utilmy/utils.py
-------------------------functions----------------------
test0()
test1()
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy/data.py
-------------------------functions----------------------
log(*s)
help()



utilmy/iio.py


utilmy/text.py
-------------------------functions----------------------
test()
log(*s)
help()
help_get_codesource(func)
test()
test_lsh()
pd_text_hash_create_lsh(df, col, sep = " ", threshold = 0.7, num_perm = 10, npool = 1, chunk  =  20000)
pd_text_getcluster(df:pd.DataFrame, col:str = 'col', threshold = 0.5, num_perm:int = 5, npool = 1, chunk  =  100000)
pd_text_similarity(df: pd.DataFrame, cols = [], algo = '')



utilmy/images/util_image.py
-------------------------functions----------------------
log(*s)
deps()
read_image(filepath_or_buffer: typing.Union[str, io.BytesIO])
visualize_in_row(**images)
maintain_aspect_ratio_resize(image, width = None, height = None, inter = cv2.INTER_AREA)



utilmy/docs/code_parser.py
-------------------------functions----------------------
export_stats_pertype(in_path:str = None, type:str = None, out_path:str = None)
export_stats_perfile(in_path:str = None, out_path:str = None)
export_stats_perrepo_txt(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_perrepo(in_path:str = None, out_path:str = None, repo_name:str = None)
export_stats_repolink_txt(repo_link: str, out_path:str = None)
export_stats_repolink(repo_link: str, out_path:str = None)
export_call_graph_url(repo_link: str, out_path:str = None)
export_call_graph(repo_link: str, out_path:str = None)
get_list_function_name(file_path)
get_list_class_name(file_path)
get_list_class_methods(file_path)
get_list_variable_global(file_path)
_get_docs(all_lines, index_1, func_lines)
get_list_function_info(file_path)
get_list_class_info(file_path)
get_list_method_info(file_path)
get_list_method_stats(file_path)
get_list_class_stats(file_path)
get_list_function_stats(file_path)
get_stats(df:pd.DataFrame, file_path:str)
get_file_stats(file_path)
get_list_imported_func(file_path: str)
get_list_import_class_as(file_path: str)
_get_words(row)
_get_functions(row)
_get_avg_char_per_word(row)
_validate_file(file_path)
_clean_data(array)
_remove_empty_line(line)
_remmove_commemt_line(line)
_get_and_clean_all_lines(file_path)
_get_all_line(file_path)
_get_all_lines_in_function(function_name, array, indentMethod = '')
_get_all_lines_in_class(class_name, array)
_get_all_lines_define_function(function_name, array, indentMethod = '')
_get_define_function_stats(array)
_get_function_stats(array, indent)
write_to_file(uri, type, list_functions, list_classes, list_imported, dict_functions, list_class_as, out_path)
test_example()



utilmy/docs/generate_doc.py
-------------------------functions----------------------
markdown_create_function(uri, name, type, args_name, args_type, args_value, start_line, list_docs, prefix = "")
markdown_create_file(list_info, prefix = '')
markdown_createall(dfi, prefix = "")
table_create_row(uri, name, type, start_line, list_funtions, prefix)
table_all_row(list_rows)
table_create(uri, name, type, start_line, list_funtions, prefix)
run_markdown(repo_stat_file, output = 'docs/doc_main.md', prefix="https = "https://github.com/user/repo/tree/a")
run_table(repo_stat_file, output = 'docs/doc_table.md', prefix="https = "https://github.com/user/repo/tree/a")
test()



utilmy/docs/__init__.py


utilmy/docs/test.py
-------------------------functions----------------------
log(data)
list_buy_price(start, bottom, delta)
calculateSellPrice(enter, profit)
list_sell_price(start, top, delta)
calculateBuyPrice(enter, profit)
get_list_price()
trading_up()
trading_down()
update_price()



utilmy/docs/cli.py
-------------------------functions----------------------
os_remove(filepath)
run_cli()



utilmy/deeplearning/zz_utils_dl2.py
-------------------------functions----------------------
np_remove_duplicates(seq)
clean1(ll)
log(*s)
log3(*s)
log2(*s)
prepro_image(image_path)
prepro_images(image_paths, nmax = 10000000)
image_center_crop(img, dim)
image_resize_pad(img, size = (256, 256)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
label_get_data()
pd_category_filter(df, category_map)
image_load(pathi, mode = 'cache')
data_add_onehot(dfref, img_dir, labels_col)
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
image_resize(img, size = (256, 256)
image_resize2(image, width = None, height = None, inter = cv2.INTER_AREA)
image_check(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_remove_bg(in_dir = "", out_dir = "", level = 1)
image_create_cache()
os_path_check(path, n = 5)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_text_blank(in_dir, out_dir, level = "/*")
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_save(out_dir)
create_train_npz()
create_train_parquet()
model_deletes(dry = 0)
topk_predict()
topk()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_export()
data_get_sample(batch_size, x_train, labels_val)
data_to_y_onehot_list(df, dfref, labels_col)
test()
unzip(in_dir, out_dir)
gzip()
folder_size()
down_ichiba()
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)
config_save(cc, path)
os_path_copy(in_dir, path, ext = "*.py")
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
train_stop(counter, patience)
model_reload(model_reload_name, cc, )
learning_rate_schedule(mode = "step", epoch = 1, cc = None)
loss_schedule(mode = "step", epoch = 1)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)
make_encoder(n_outputs = 1)
make_decoder()
make_classifier(class_dict)
predict(name = None)
metric_accuracy_test(y_test, y_pred, dd)
metric_accuracy_val(y_val, y_pred_head, class_dict)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
metric_accuracy2(y_test, y_pred, dd)
clf_loss_macro_soft_f1(y, y_hat)
check_tf()
gpu_usage()
gpu_free()
train_step(x, model, y_label_list = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
RealCustomDataGenerator.__init__(self, image_dir, label_path, class_dict, split = 'train', batch_size = 8, transforms = None, shuffle = False, img_suffix = ".png")
RealCustomDataGenerator._load_data(self, label_path)
RealCustomDataGenerator.on_epoch_end(self)
RealCustomDataGenerator.__len__(self)
RealCustomDataGenerator.__getitem__(self, idx)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.on_epoch_end(self)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.__getitem__(self, idx)
SprinklesTransform.__init__(self, num_holes = 100, side_length = 10, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)
StepDecay.__call__(self, epoch)
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list =  None)


utilmy/deeplearning/__init__.py


utilmy/deeplearning/util_topk.py
-------------------------functions----------------------
topk_predict()
topk()
topk_nearest_vector(x0, vector_list, topk = 3)
topk_export()



utilmy/deeplearning/zz_prepro.py
-------------------------functions----------------------
log(*s)
prepro_images(image_paths, nmax = 10000000)
prepro_image0(image_path)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
run_multiprocess(myfun, list_args, npool = 10, **kwargs)
create_train_npz()
image_resize(out_dir = "")
image_check()
create_train_parquet()
model_deletes(dry = 0)
data_add_onehot(dfref, img_dir, labels_col)
test()
unzip(in_dir, out_dir)
gzip()
predict(name = None)
folder_size()
prepro_images2(image_paths)



utilmy/deeplearning/util_image.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
test()
prepro_image(image_path:str, xdim = 1, ydim = 1)
prepro_images(image_paths, nmax = 10000000)
image_center_crop(img, dim)
prepro_images_multi(image_paths, npool = 30, prepro_image = None)
image_resize_pad(img, size = (256, 256)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_save_tocache(out_dir, name = "cache1")
image_check_npz(path_npz, keys = ['train'], path = "", tag = "", n_sample = 3, renorm = True)
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_center_crop(img, dim)
image_resize_pad(img, size = (256, 256)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(img, size = (256, 256)
image_read(filepath_or_buffer: Union[str, io.BytesIO])
image_remove_bg(in_dir = "", out_dir = "", level = 1)
image_create_cache()
os_path_check(path, n = 5)
image_face_blank(in_dir = "", level  =  "/*", out_dir = f"", npool = 30)
image_text_blank(in_dir, out_dir, level = "/*")
image_save(out_dir, name = "cache1")



utilmy/deeplearning/util_dl.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
test()
tensorboard_log(pars_dict:dict = None, writer = None, verbose = True)
tf_check()
gpu_usage()
gpu_free()
down_page(query, out_dir = "query1", genre_en = '', id0 = "", cat = "", npage = 1)



utilmy/configs/__init__.py


utilmy/configs/test.py
-------------------------functions----------------------
create_fixtures_data(tmp_path)
test_validate_yaml_types(tmp_path)
test_validate_yaml_types_failed(tmp_path)
test_validate_yaml_failed_silent(tmp_path)



utilmy/configs/util_config.py
-------------------------functions----------------------
log(*s)
loge(*s)
test_yamlschema()
test_pydanticgenrator()
test4()
test_example()
config_load(config_path:    str   =  None, path_default:   str   =  None, config_default: dict  =  None, save_default:   bool  =  False, to_dataclass:   bool  =  True, )
config_isvalid_yamlschema(config_dict: dict, schema_path: str  =  'config_val.yaml', silent: bool  =  False)
config_isvalid_pydantic(config_dict: dict, pydanctic_schema: str  =  'config_py.yaml', silent: bool  =  False)
convert_yaml_to_box(yaml_path: str)
convert_dict_to_pydantic(config_dict: dict, schema_name: str)
pydantic_model_generator(input_file: Union[Path, str], input_file_type, output_file: Path, **kwargs, )
global_verbosity(cur_path, path_relative = "/../../config.json", default = 5, key = 'verbosity', )
zzz_config_load_validate(config_path: str, schema_path: str, silent: bool  =  False)



utilmy/viz/embedding.py
-------------------------functions----------------------
log(*s)
embedding_load_word2vec(model_vector_path = "model.vec", nmax  =  500)
embedding_load_parquet(path = "df.parquet", nmax  =  500)
tokenize_text(text)
run(dir_in = "in/model.vec", dir_out = "ztmp/", nmax = 100)

-------------------------methods----------------------
vizEmbedding.__init__(self, path = "myembed.parquet", num_clusters = 5, sep = ";", config:dict = None)
vizEmbedding.run_all(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = "ztmp/")
vizEmbedding.dim_reduction(self, mode = "mds", col_embed = 'embed', ndim = 2, nmax =  5000, dir_out = None)
vizEmbedding.create_clusters(self, after_dim_reduction = True)
vizEmbedding.create_visualization(self, dir_out = "ztmp/", mode = 'd3', cols_label = None, show_server = False, **kw)
vizEmbedding.draw_hiearchy(self)


utilmy/viz/util_map.py


utilmy/viz/__init__.py


utilmy/viz/vizhtml.py
-------------------------functions----------------------
test1(verbose = False)
test2(verbose = False)
mlpd3_add_tooltip(fig, points, labels)
pd_plot_scatter_get_data(df0:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, nmax: int = 20000, **kw)
pd_plot_scatter_matplot(df:pd.DataFrame, colx: str = None, coly: str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, cfg: dict  =  {}, mode = 'd3', save_path: str = '', verbose = True, **kw)
pd_plot_histogram_matplot(df:pd.DataFrame, col: str = '', colormap:str = 'RdYlBu', title: str = '', nbin = 20.0, q5 = 0.005, q95 = 0.995, nsample = -1, save_img: str = "", xlabel: str = None, ylabel: str = None, verbose = True, **kw)
pd_plot_tseries_matplot(df:pd.DataFrame, plot_type: str = None, coly1: list  =  [], coly2: list  =  [], 8, 4), spacing = 0.1, verbose = True, **kw))
mpld3_server_start()
pd_plot_highcharts(df)
pd_plot_scatter_highcharts(df0:pd.DataFrame, colx:str = None, coly:str = None, collabel: str = None, colclass1: str = None, colclass2: str = None, colclass3: str = None, nsample = 10000, cfg:dict = {}, mode = 'd3', save_img = '', verbose = True, **kw)
pd_plot_tseries_highcharts(df, coldate:str = None, date_format:str = '%m/%d/%Y', coly1:list  = [], coly2:list  = [], figsize:tuple  =   None, title:str = None, xlabel:str = None, y1label:str = None, y2label:str = None, cfg:dict = {}, mode = 'd3', save_img = "", verbose = True, **kw)
pd_plot_histogram_highcharts(df:pd.DataFrame, colname:str = None, binsNumber = None, binWidth = None, title:str = "", xaxis_label:str =  "x-axis", yaxis_label:str = "y-axis", cfg:dict = {}, mode = 'd3', save_img = "", show = False, verbose = True, **kw)
html_show_chart_highchart(html_code, verbose = True)
html_show(html_code, verbose = True)
images_to_html(dir_input = "*.png", title = "", verbose = False)
colormap_get_names()
pd_plot_network(df:pd.DataFrame, cola: str = 'col_node1', colb: str = 'col_node2', coledge: str = 'col_edge', colweight: str = "weight", html_code:bool  =  True)
help_get_codesource(func)
zz_css_get_template(css_name:str =  "A4_size")
zz_test_get_random_data(n = 100)
zz_pd_plot_histogram_highcharts_old(df, col, figsize = None, title = None, cfg:dict = {}, mode = 'd3', save_img = '')

-------------------------methods----------------------
mpld3_TopToolbar.__init__(self)


utilmy/tseries/util_tseries.py


utilmy/logs/test_log.py
-------------------------functions----------------------
test1()
test2()
test_launch_server()
test_server()

-------------------------methods----------------------
LoggingStreamHandler.handle(self)


utilmy/logs/__init__.py


utilmy/logs/util_log.py
-------------------------functions----------------------
logger_setup(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
log(log_config_path: str  =  None, log_template: str  =  "default", **kwargs)
log2(*s)
log3(*s)
logw(*s)
logc(*s)
loge(*s)
logr(*s)
test()
z_logger_stdout_override()
z_logger_custom_1()



utilmy/spark/setup.py


utilmy/spark/main.py
-------------------------functions----------------------
spark_init(config:dict)
main()



utilmy/excel/xlvba.py
-------------------------functions----------------------
load_csv(csvfile)
invokenumpy()
invokesklearn()
loaddf()



utilmy/templates/__init__.py


utilmy/templates/cli.py
-------------------------functions----------------------
run_cli()
template_show()
template_copy(name, out_dir)



utilmy/zarchive/_HELP.py
-------------------------functions----------------------
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
os_VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)
os_VS_start(self, version)
fun_cython(a)
fun_python(a)



utilmy/zarchive/global01.py


utilmy/zarchive/excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy/zarchive/kagglegym.py
-------------------------functions----------------------
r_score(y_true, y_pred, sample_weight = None, multioutput = None)
make()

-------------------------methods----------------------
Observation.__init__(self, train, target, features)
Environment.__init__(self)
Environment.reset(self)
Environment.step(self, target)
Environment.__str__(self)


utilmy/zarchive/fast.py
-------------------------functions----------------------
day(s)
month(s)
year(s)
hour(s)
weekday(s)
season(d)
daytime(d)
fastStrptime(val, format)
drawdown_calc_fast(price)
std(x)
mean(x)
_compute_overlaps(u, v)
distance_jaccard2(u, v)
distance_jaccard(u, v)
distance_jaccard_X(X)
cosine(u, v)
rmse(y, yhat)
cross(vec1, vec2)
norm(vec)
log_exp_sum2(a, b)



utilmy/zarchive/allmodule_fin.py


utilmy/zarchive/util_web.py
-------------------------functions----------------------
web_restapi_toresp(apiurl1)
web_getrawhtml(url1)
web_importio_todataframe(apiurl1, isurl = 1)
web_getjson_fromurl(url)
web_gettext_fromurl(url, htmltag = 'p')
web_gettext_fromhtml(file1, htmltag = 'p')
web_getlink_fromurl(url)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)



utilmy/zarchive/datanalysis.py
-------------------------functions----------------------
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_describe(df)
pd_stack_dflist(df_list)
pd_validation_struct()
pd_checkpoint()
xl_setstyle(file1)
xl_val(ws, colj, rowi)
isnull(x)
xl_get_rowcol(ws, i0, j0, imax, jmax)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
str_to_unicode(x, encoding = 'utf-8')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, header = True, maxline = -1)
csv_analysis()
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', nrow = 1000000, chunk =  5000000)
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_bigcompute()
db_getdata()
db_sql()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
col_study_getcategorydict_freq(catedict)
col_feature_importance(Xcol, Ytarget)
pd_col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
pd_col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, do_plot = 1, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b', annotate_above = 0)
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
sk_catboost_classifier(Xtrain, Ytrain, Xcolname = None, pars= {"learning_rate" =  {"learning_rate":0.1, "iterations":1000, "random_seed":0, "loss_function": "MultiClass" }, isprint = 0)
sk_catboost_regressor()
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_feature_importance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy/zarchive/alldata.py


utilmy/zarchive/report.py
-------------------------functions----------------------
map_show()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)
xl_create_pdf()



utilmy/zarchive/linux.py
-------------------------functions----------------------
load_session(name = 'test_20160815')
save_session(name = '')
isfloat(value)
isint(x)
aa_isanaconda()
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
np_interpolate_nan(y)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
sortcol(arr, colid, asc = 1)
sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_memory_array_adress(x)
sk_featureimportance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_date_intersection(qlist)
pd_resetindex(df)
pd_create_colmap_nametoid(df)
pd_dataframe_toarray(df)
pd_changeencoding(data, cols)
pd_createdf(val1, col1 = None, idx1 = None)
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_replacevalues(df, matrix)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_remove_row(df, row_list_index = [23, 45])
pd_extract_col_idx_val(df)
pd_split_col_idx_val(df)
pd_addcolumn(df1, name1 = 'new')
pd_removecolumn(df1, name1)
pd_save_vectopanda(vv, filenameh5)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
dateint_todatetime(datelist1)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
textvect_topanda(vv, fileout = "")
comoment(xx, yy, nsample, kx, ky)
acf(data)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
VS_start(self, version)
VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)



utilmy/zarchive/portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_finddateid(date1, dateref)
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy/zarchive/util.py
-------------------------functions----------------------
session_load_function(name = 'test_20160815')
session_save_function(name = 'test')
py_save_obj_dill(obj1, keyname = '', otherfolder = 0)
session_spyder_showall()
session_guispyder_save(filename)
session_guispyder_load(filename)
session_load(name = 'test_20160815')
session_save(name = 'test')
aa_unicode_ascii_utf8_issue()
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
a_run_ipython(cmd1)
a_autoreload()
a_start_log(id1 = '', folder = 'aaserialize/log/')
a_cleanmemory()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_info_conda_jupyter()
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = 'a')
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_isame(file1, file2)
os_file_get_file_extension(file_path)
os_file_normpath(path)
os_folder_is_path(path_or_stream)
os_file_get_path_from_stream(maybe_stream)
os_file_try_to_get_extension(path_or_strm)
os_file_are_same_file_types(paths)
os_file_norm_paths(paths, marker = '*')
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_wait_cpu(priority = 300, cpu_min = 50)
os_split_dir_file(dirfile)
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj1, keyname = '', otherfolder = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)
sql_getdate()
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
print_topdf()
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_config_getfile(file1)
os_csv_process(file1)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input, errors = 'replace')
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_remove_NA_INF_2d(X)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = [])
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
min_kpos(arr, kth)
max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes(df, columns = [], targetype = 'category')
pd_df_todict2(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_df_todict(df1, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace =  True)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = [23, 45])
pd_removecol(df1, name1)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
date_allinfo()
date_convert(t1, fromtype, totype)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_diffinbday(intd2, intd1)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_tofloat(dt)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_comoment(xx, yy, nsample, kx, ky)
np_acf(data)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
np_map_dict_to_bq_schema(source_dict, schema, dest_dict)
googledrive_get()
googledrive_put()
googledrive_list()
os_processify_fun(func)
ztest_processify()
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
py_exception_print()
py_log_write(LOGFILE, prefix)

-------------------------methods----------------------
testclass.__init__(self, x)
testclass.z_autotest(self)
FundingRecord.parse(klass, row)
FundingRecord.__str__(self)


utilmy/zarchive/multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy/zarchive/fast_parallel.py
-------------------------functions----------------------
task_summary(tasks)
task_progress(tasks)
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)



utilmy/zarchive/util_min.py
-------------------------functions----------------------
os_wait_cpu(priority = 300, cpu_min = 50)
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_print_tofile(vv, file1, mode1 = 'a')
a_get_pythonversion()
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)



utilmy/zarchive/function_custom.py
-------------------------functions----------------------
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
getweight(ww, size = (9, 3)
fun_obj(vv, ext)



utilmy/zarchive/__init__.py


utilmy/zarchive/util_aws.py
-------------------------functions----------------------
aws_credentials(account = None)
aws_ec2_get_instanceid(con, ip_address)
aws_ec2_allocate_elastic_ip(con, instance_id = "", elastic_ip = '', region = "ap-northeast-2")
aws_ec2_printinfo(instance = None, ipadress = "", instance_id = "")
aws_ec2_spot_start(con, region, key_name = "ecsInstanceRole", inst_type = "cx2.2", ami_id = "", pricemax = 0.15, elastic_ip = '', pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_get_id(ipadress = '', instance_id = '')
aws_ec2_spot_stop(con, ipadress = "", instance_id = "")
aws_ec2_res_start(con, region, key_name, ami_id, inst_type = "cx2.2", min_count  = 1, max_count  = 1, pars= {"security_group" =  {"security_group": [""], "disk_size": 25, "disk_type": "ssd", "volume_type": "gp2"})
aws_ec2_res_stop(con, ipadress = "", instance_id = "")
aws_accesskey_get(access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_getinfo(conn)
aws_s3_url_split(url)
aws_s3_getbucketconn(s3dir)
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_file_read(bucket1, filepath, isbinary = 1)
aws_ec2_cmd_ssh(cmdlist =   ["ls " ], host = 'ip', doreturn = 0, ssh = None, username = 'ubuntu', keyfilepath = '')
aws_ec2_python_script(script_path, args1, host)
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
ztest_01()

-------------------------methods----------------------
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.write_command(self, text, remotefile)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh._help_ssh(self)


utilmy/zarchive/coke_functions.py
-------------------------functions----------------------
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
date_diffend(t)
np_dict_tolist(dd)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
day(s)
month(s)
year(s)
hour(s)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
season(d)
daytime(d)
pd_date_splitall(df, coldate = 'purchased_at')



utilmy/zarchive/allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')



utilmy/zarchive/multithread.py
-------------------------functions----------------------
multithread_run(fun_async, input_list:list, n_pool = 5, start_delay = 0.1, verbose = True, **kw)
multithread_run_list(**kwargs)



utilmy/zarchive/geospatial.py


utilmy/zarchive/portfolio_withdate.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(tt)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
date_todatetime(tlist)
date_removetimezone(datelist)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy/zarchive/filelock.py
-------------------------methods----------------------
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.locked(self)
FileLock.available(self)
FileLock.acquire(self, blocking = True)
FileLock.release(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__del__(self)
FileLock.purge(self)


utilmy/zarchive/util_ml.py
-------------------------functions----------------------
create_weight_variable(name, shape)
create_bias_variable(name, shape)
create_adam_optimizer(learning_rate, momentum)
tf_check()
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.create_batches(self)
TextLoader.next_batch(self)
TextLoader.reset_batch_pointer(self)


utilmy/zarchive/utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy/zarchive/rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy/zarchive/util_sql.py
-------------------------functions----------------------
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_delete_table(name, dbengine)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_postgres_pivot()
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')



utilmy/deeplearning/torch/util_train.py


utilmy/deeplearning/keras/template_train.py
-------------------------functions----------------------
param_set()
params_set2()
np_remove_duplicates(seq)
clean1(ll)
log3(*s)
log2(*s)
log(*s)
config_save(cc, path)
os_path_copy(in_dir, path, ext = "*.py")
metric_accuracy(y_val, y_pred_head, class_dict)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
train_stop(counter, patience)
model_reload(model_reload_name, cc, )
image_check(name, img, renorm = False)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
label_get_data()
pd_category_filter(df, category_map)
train_step(x, model, y_label_list = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)


utilmy/deeplearning/keras/util_layers.py
-------------------------functions----------------------
log(*s)
log2(*s)
help()
make_encoder(n_outputs = 1)
make_decoder()
make_classifier(class_dict)

-------------------------methods----------------------
DFC_VAE.__init__(self, latent_dim, class_dict)
DFC_VAE.encode(self, x)
DFC_VAE.reparameterize(self, z_mean, z_logsigma)
DFC_VAE.decode(self, z, apply_sigmoid = False)
DFC_VAE.call(self, x, training = True, mask = None, y_label_list =  None)


utilmy/deeplearning/keras/util_train.py
-------------------------functions----------------------
np_remove_duplicates(seq)
clean1(ll)
log3(*s)
log2(*s)
tf_compute_set(cc:dict)
log(*s)
config_save(cc, path)
os_path_copy(in_dir, path, ext = "*.py")
metric_accuracy(y_val, y_pred_head, class_dict)
valid_image_original(img_list, path, tag, y_labels, n_sample = None)
valid_image_check(img_list, path = "", tag = "", y_labels = "", n_sample = 3, renorm = True)
save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd)
save_model_state(model, model_dir2)
train_stop(counter, patience)
model_reload(model_reload_name, cc, )
image_check(name, img, renorm = False)
pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot = True)
label_get_data()
pd_category_filter(df, category_map)
train_step(x, model, y_label_list = None)
validation_step(x, model, y_label_list = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)


utilmy/deeplearning/keras/util_loss.py
-------------------------functions----------------------
log(*s)
metric_accuracy(y_test, y_pred, dd)
clf_loss_macro_soft_f1(y, y_hat)
learning_rate_schedule(mode = "step", epoch = 1, cc = None)
loss_schedule(mode = "step", epoch = 1)
perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight = 0.00005, y_label_heads = None, y_pred_heads = None, clf_loss_fn = None)

-------------------------methods----------------------
LearningRateDecay.plot(self, epochs, title = "Learning Rate Schedule", path = None)
StepDecay.__init__(self, init_lr = 0.01, factor = 0.25, drop_every = 5)
StepDecay.__call__(self, epoch)


utilmy/deeplearning/keras/util_dataloader.py
-------------------------functions----------------------
log(*s)
data_get_sample(batch_size, x_train, labels_val)
data_to_y_onehot_list(df, dfref, labels_col)
data_add_onehot(dfref, img_dir, labels_col)

-------------------------methods----------------------
CustomDataGenerator.__init__(self, x, y, batch_size = 32, augmentations = None)
CustomDataGenerator.__len__(self)
CustomDataGenerator.__getitem__(self, idx)
CustomDataGenerator_img.__init__(self, img_dir, label_path, class_list, split = 'train', batch_size = 8, transforms = None)
CustomDataGenerator_img.on_epoch_end(self)
CustomDataGenerator_img.__len__(self)
CustomDataGenerator_img.__getitem__(self, idx)
SprinklesTransform.__init__(self, num_holes = 30, side_length = 5, always_apply = False, p = 1.0)
SprinklesTransform.apply(self, image, **params)


utilmy/viz/zarchive/__init__.py


utilmy/viz/zarchive/toptoolbar.py
-------------------------methods----------------------
TopToolbar.__init__(self)


utilmy/spark/src/util_models.py
-------------------------functions----------------------
TimeSeriesSplit(df_m:pyspark.sql.DataFrame, splitRatio:float, sparksession:object)
Train(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
Predict(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str = None, conf_model:dict = None)
os_makedirs(path:str)



utilmy/spark/src/__init__.py


utilmy/spark/src/utils.py
-------------------------functions----------------------
logger_setdefault()
log()
log2(*s)
log3(*s)
log()
log_sample(*s)
config_load(config_path:str)
spark_check(df:pyspark.sql.DataFrame, conf:dict = None, path:str = "", nsample:int = 10, save = True, verbose = True, returnval = False)

-------------------------methods----------------------
to_namespace.__init__(self, d)


utilmy/spark/script/hadoopVersion.py


utilmy/spark/script/pysparkTest.py
-------------------------functions----------------------
inside(p)



utilmy/spark/tests/test_functions.py
-------------------------functions----------------------
test_getall_families_from_useragent(spark_session: SparkSession)



utilmy/spark/tests/__init__.py


utilmy/spark/tests/test_table_user_session_log.py
-------------------------functions----------------------
test_table_user_session_log_run(spark_session: SparkSession)
test_table_user_session_log(spark_session: SparkSession)
test_table_usersession_log_stats(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_table_volume_predict.py
-------------------------functions----------------------
test_preprocess(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_table_user_session_stats.py
-------------------------functions----------------------
test_table_user_session_stats_run(spark_session: SparkSession)
test_table_user_session_stats(spark_session: SparkSession)
test_table_user_session_stats_ip(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_common.py
-------------------------functions----------------------
assert_equal_spark_df_sorted(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str)
assert_equal_spark_df_schema(expected_schema: [tuple], actual_schema: [tuple], df_name: str)



utilmy/spark/tests/conftest.py
-------------------------functions----------------------
config()
spark_session(config: dict)



utilmy/spark/tests/test_utils.py
-------------------------functions----------------------
test_spark_check(spark_session: SparkSession, config: dict)



utilmy/spark/tests/test_table_user_log.py
-------------------------functions----------------------
test_table_user_log_run(spark_session: SparkSession, config: dict)



utilmy/zarchive/py3/util.py
-------------------------functions----------------------
session_load_function(name = 'test_20160815')
session_save_function(name = 'test')
py_save_obj_dill(obj1, keyname)
session_spyder_showall()
session_guispyder_save(filename)
session_guispyder_load(filename)
session_load(name = 'test_20160815')
session_save(name = 'test')
aa_unicode_ascii_utf8_issue()
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
a_run_ipython(cmd1)
a_autoreload()
a_get_platform()
a_start_log(id1 = '', folder = 'aaserialize/log/')
a_cleanmemory()
a_module_codesample(module_str = 'pandas')
a_module_doc(module_str = 'pandas')
a_module_generatedoc(module_str = "pandas", fileout = '')
a_info_conda_jupyter()
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = 'a')
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_extracttext_allfile(nfile, dir1, pattern1 = "*.html", htmltag = 'p', deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
os_process_run(cmd_list = ['program', 'arg1', 'arg2'], capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj1, keyname)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)
sql_getdate()
obj_getclass_of_method(meth)
obj_getclass_property(pfi)
print_topdf()
os_config_setfile(dict_params, outfile, mode1 = 'w+')
os_config_getfile(file1)
os_csv_process(file1)
read_funding_data(path)
read_funding_data(path)
read_funding_data(path)
find_fuzzy(xstring, list_string)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input, errors = 'replace')
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, numSpaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')
web_restapi_toresp(apiurl1)
web_getrawhtml(url1)
web_importio_todataframe(apiurl1, isurl = 1)
web_getjson_fromurl(url)
web_gettext_fromurl(url, htmltag = 'p')
web_gettext_fromhtml(file1, htmltag = 'p')
web_getlink_fromurl(url)
web_send_email(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_send_email_tls(FROM, recipient, subject, body, login1 = "mizenjapan@gmail.com", pss1 = "sophieelise237", server1 = "smtp.gmail.com", port1 = 465)
web_sendurl(url1)
np_minimize(fun_obj, x0 = [0.0], argext = (0, 0)
np_minimizeDE(fun_obj, bounds, name1, solver = None)
np_remove_NA_INF_2d(X)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = [])
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = [])
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
min_kpos(arr, kth)
max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(xstring, list_string)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
sql_create_dbengine(type1 = '', dbname = '', login = '', password = '', url = 'localhost', port = 5432)
sql_query(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', dbengine = None, output = 'df', dburl='sqlite = 'sqlite:///aaserialize/store/finviz.db')
sql_get_dbschema(dburl='sqlite = 'sqlite:///aapackage/store/yahoo.db', dbengine = None, isprint = 0)
sql_delete_table(name, dbengine)
sql_insert_excel(file1 = '.xls', dbengine = None, dbtype = '')
sql_insert_df(df, dbtable, dbengine, col_drop = ['id'], verbose = 1)
sql_insert_csv(csvfile, dbtable, dbengine, col_drop = [])
sql_insert_csv2(csvfile = '', dbtable = '', columns = [], dbengine = None, nrows =  10000)
sql_postgres_create_table(mytable = '', database = '', username = '', password = '')
sql_postgres_query_to_csv(sqlr = 'SELECT ticker,shortratio,sector1_id, FROM stockfundamental', csv_out = '')
sql_postgres_pivot()
sql_mysql_insert_excel()
sql_pivotable(dbcon, ss = 'select  ')
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_selectrow(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = '', csvfile = '')
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = '*', col_restrict = [], isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = [], targetype = 'category')
pd_dtypes(df, columns = [], targetype = 'category')
pd_df_todict(df, colkey = 'table', excludekey = [''], onlyfirstelt =  True)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_cleanquote(q)
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = 'iso-8859-1', toenc = 'utf-8')
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df_insider, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = [23, 45])
pd_removecol(df1, name1)
pd_addcol(df1, name1 = 'new')
pd_insertcol(df, colname, vec)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile='F = 'F:\temp_pandas.h5')
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_h5_save(df, filenameh5='E = 'E:/_data/_data_outlier.h5', key = 'data')
pd_h5_load(filenameh5='E = 'E:/_data/_data_outlier.h5', table_id = 'data', exportype = "pandas", rowstart = -1, rowend = -1, cols = [])
pd_h5_fromcsv_tohdfs(dircsv = 'dir1/dir2/', filepattern = '*.csv', tofilehdfs = 'file1.h5', tablename = 'df', col_category = [], dtype0 = None, encoding = 'utf-8', chunksize =  2000000, mode = 'a', format = 'table', complib = None)
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = 'data')
date_allinfo()
date_convert(t1, fromtype, totype)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinyear(startdate, enddate)
date_diffinbday(intd2, intd1)
date_gencalendar(start = '2010-01-01', end = '2010-01-15', country = 'us')
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = 'str', format1= "%Y-%m-%d %H =  "%Y-%m-%d %H:%M:%S:%f")
date_tofloat(dt)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_numexpr_tohdfs(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
np_comoment(xx, yy, nsample, kx, ky)
np_acf(data)
plot_XY(xx, yy, zcolor = None, tsize = None, title1 = '', xlabel = '', ylabel = '', figsize = (8, 6)
plot_heatmap(frame, ax = None, cmap = None, vmin = None, vmax = None, interpolation = 'nearest')
gc_map_dict_to_bq_schema(source_dict, schema, dest_dict)
aws_accesskey_get(access = '', key = '')
aws_conn_do(action = '', region = "ap-northeast-2")
aws_conn_getallregions(conn = None)
aws_conn_create(region = "ap-northeast-2", access = '', key = '')
aws_conn_getinfo(conn)
aws_s3_url_split(url)
aws_s3_getbucketconn(s3dir)
aws_s3_puto_s3(fromdir_file = 'dir/file.zip', todir = 'bucket/folder1/folder2')
aws_s3_getfrom_s3(froms3dir = 'task01/', todir = '', bucket_name = 'zdisk')
aws_s3_folder_printtall(bucket_name = 'zdisk')
aws_s3_file_read(filepath, isbinary = 1)
aws_ec2_python_script(script_path, args1, host)
aws_ec2_create_con(contype = 'sftp/ssh', host = 'ip', port = 22, username = 'ubuntu', keyfilepath = '', password = '', keyfiletype = 'RSA', isprint = 1)
aws_ec2_allocate_elastic_ip(instance_id, region = "ap-northeast-2")
googledrive_get()
googledrive_put()
googledrive_list()
os_processify_fun(func)
ztest_processify()
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )

-------------------------methods----------------------
testclass.__init__(self, x)
testclass.z_autotest(self)
FundingRecord.parse(klass, row)
FundingRecord.__str__(self)
aws_ec2_ssh.__init__(self, hostname, username = 'ubuntu', key_file = None, password = None)
aws_ec2_ssh.command(self, cmd)
aws_ec2_ssh.put(self, localfile, remotefile)
aws_ec2_ssh.put_all(self, localpath, remotepath)
aws_ec2_ssh.get(self, remotefile, localfile)
aws_ec2_ssh.sftp_walk(self, remotepath)
aws_ec2_ssh.get_all(self, remotepath, localpath)
aws_ec2_ssh.write_command(self, text, remotefile)
aws_ec2_ssh.python_script(self, script_path, args1)
aws_ec2_ssh.command_list(self, cmdlist)
aws_ec2_ssh.listdir(self, remotedir)
aws_ec2_ssh.jupyter_kill(self)
aws_ec2_ssh.jupyter_start(self)
aws_ec2_ssh.cmd2(self, cmd1)
aws_ec2_ssh._help_ssh(self)


utilmy/zarchive/storage/global01.py


utilmy/zarchive/storage/theano_lstm.py
-------------------------functions----------------------
numpy_floatX(data)
get_minibatches_idx(n, minibatch_size, shuffle = False)
get_dataset(name)
zipp(params, tparams)
unzip(zipped)
dropout_layer(state_before, use_noise, trng)
_p(pp, name)
init_params(options)
load_params(path, params)
init_tparams(params)
get_layer(name)
ortho_weight(ndim)
param_init_lstm(options, params, prefix = 'lstm')
lstm_layer(tparams, state_below, options, prefix = 'lstm', mask = None)
sgd(lr, tparams, grads, x, mask, y, cost)
adadelta(lr, tparams, grads, x, mask, y, cost)
rmsprop(lr, tparams, grads, x, mask, y, cost)
build_model(tparams, options)
pred_probs(f_pred_prob, prepare_data, data, iterator, verbose = False)
pred_error(f_pred, prepare_data, data, iterator, verbose = False)
train_lstm(dim_proj = 128, # word embeding dimension and LSTM number of hidden units.patience = 10, # Number of epoch to wait before early stop if no progressmax_epochs = 5000, # The maximum number of epoch to rundispFreq = 10, # Display to stdout the training progress every N updatesdecay_c = 0., # Weight decay for the classifier applied to the U weights.not used for adadelta and rmsprop)n_words = 10000, # Vocabulary sizeprobably need momentum and decaying learning rate).encoder = 'lstm', # TODO: can be removed must be lstm.saveto = 'lstm_model.npz', # The best model will be saved therevalidFreq = 370, # Compute the validation error after this number of update.saveFreq = 1110, # Save the parameters after every saveFreq updatesmaxlen = 100, # Sequence longer then this get ignoredbatch_size = 16, # The batch size during training.valid_batch_size = 64, # The batch size used for validation/test set.dataset = 'imdb', noise_std = 0., use_dropout = True, # if False slightly faster, but worst test errorreload_model = None, # Path to a saved model we want to start from.test_size = -1, # If >0, we keep only this number of test example.)



utilmy/zarchive/storage/excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy/zarchive/storage/theano_imdb.py
-------------------------functions----------------------
prepare_data(seqs, labels, maxlen = None)
get_dataset_file(dataset, default_dataset, origin)
load_data(path = "imdb.pkl", n_words = 100000, valid_portion = 0.1, maxlen = None, sort_by_len = True)



utilmy/zarchive/storage/rec_data.py
-------------------------functions----------------------
_get_movielens_path()
_download_movielens(dest_path)
_get_raw_movielens_data()
_parse(data)
_build_interaction_matrix(rows, cols, data)
_get_movie_raw_metadata()
get_movielens_item_metadata(use_item_ids)
get_dense_triplets(uids, pids, nids, num_users, num_items)
get_triplets(mat)
get_movielens_data()



utilmy/zarchive/storage/java.py
-------------------------functions----------------------
importJAR(path1 = "", path2 = "", path3 = "", path4 = "")
listallfile(some_dir, pattern = "*.*", dirlevel = 1)
importFolderJAR(dir1 = "", dirlevel = 1)
importFromMaven()
showLoadedClass()
inspectJAR(dir1)
loadSingleton(class1)
java_print(x)
compileJAVA(javafile)
writeText(text, filename)
compileJAVAtext(classname, javatxt, path1 = "")
execute_javamain(java_file)
javaerror(jpJavaException)
launchPDFbox()
getfpdffulltext(pdfile1)
launchTIKA()
getfulltext(file1, withMeta = 0)
directorygetalltext(dir1, filetype1 = "*.*", withMeta = 0, fileout = "")
directorygetalltext2(dir1, filetype1 = "*.*", type1 = 0, fileout = "")



utilmy/zarchive/storage/alldata.py


utilmy/zarchive/storage/portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(matx, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float16)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_lowbandtrend1(close2, type1 = 0)
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_inverseetf(price, costpa = 0.0)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref)
folio_riskpa(ret, targetvol = 0.1, volrange = 90)
objective_criteria(bsk, criteria, date1 = None)
calcbasket_obj(wwvec, *data)
calc_optimal_weight(args, bounds, maxiter = 1)
fitness(p)
np_countretsign(x)
np_trendtest(x, alpha  =  0.05)
correl_rankbystock(stkid = [2, 5, 6], correl = [[1, 0], [0, 1]])
calc_print_correlrank(close2, symjp1, nlag, refindexname, toprank2 = 5, customnameid = [], customnameid2 = [])
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
imp_findticker(tickerlist, sym01, symname)
imp_close_dateref(sym01, sdate = 20100101, edate = 20160628, datasource = '', typeprice = "close")
imp_yahooticker(symbols, start = "20150101", end = "20160101", type1 = 1)
imp_errorticker(symbols, start = "20150101", end = "20160101")
imp_yahoo_financials_url(ticker_symbol, statement = "is", quarterly = False)
imp_yahoo_periodic_figure(soup, yahoo_figure)
imp_googleIntradayQuoteSave(name1, date1, inter, tframe, dircsv)
imp_googleQuoteSave(name1, date1, date2, dircsv)
imp_googleQuoteList(symbols, date1, date2, inter = 23400, tframe = 2000, dircsv = '', intraday1 = True)
pd_filterbydate(df, dtref = None, start='2016-06-06 00 = '2016-06-06 00:00:00', end='2016-06-14 00 = '2016-06-14 00:00:00', freq = '0d0h05min', timezone = 'Japan')
imp_panda_db_dumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
imp_numpyclose_frompandas(dbfile, symlist = [], t0 = 20010101, t1 = 20010101, priceid = "close", maxasset = 2500, tmax2 = 2000)
imp_quotes_fromtxt(stocklist01, filedir='E = 'E:/_data/stock/daily/20160610/jp', startdate = 20150101, endate = 20160616)
imp_quotes_errordate(quotes, dateref)
imp_getcsvname(name1, date1, inter, tframe)
imp_quote_tohdfs(sym, qqlist, filenameh5, fromzone = 'Japan', tozone = 'UTC')
date_todatetime(tlist)
date_removetimezone(datelist)
imp_csvquote_topanda(file1, filenameh5, dfname = 'sym1', fromzone = 'Japan', tozone = 'UTC')
imp_panda_insertfoldercsv(dircsv, filepd= r'E =  r'E:\_data\stock\intraday_google.h5', fromtimezone = 'Japan', tozone = 'UTC')
imp_panda_checkquote(quotes)
imp_panda_getquote(filenameh5, dfname = "data")
imp_pd_merge_database(filepdfrom, filepdto)
imp_panda_getListquote(symbols, close1 = 'close', start='12/18/2015 00 = '12/18/2015 00:00:00+00:00', end='3/1/2016 00 = '3/1/2016 00:00:00+00:00', freq = '0d0h10min', filepd= 'E =  'E:\_data\stock\intraday_google.h5', tozone = 'Japan', fillna = True, interpo = True)
imp_panda_cleanquotes(df, datefilter)
imp_panda_storecopy()
imp_panda_removeDuplicate(filepd=  'E =   'E:\_data\stock\intraday_google.h5')
calc_statestock(close2, dateref, symfull)
imp_screening_addrecommend(string1, dbname = 'stock_recommend')
imp_finviz()
imp_finviz_news()
imp_finviz_financials()
get_price2book(symbol)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close()
index.updatehisto()
index.help()
index._statecalc(self)
index._objective_criteria(self, bsk)
index.calcbasket_obj(self, wwvec)
index.calc_optimal_weight(self, maxiter = 1)
index._weightcalc_generic(self, wwvec, t)
index._weightcalc_constant(self, ww2, t)
index._weightcalc_regime2(self, wwvec, t)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results(self, filename)
Quote.__init__(self)
Quote.append(self, dt, open_, high, low, close, volume)
Quote.to_csv(self)
Quote.write_csv(self, filename)
Quote.read_csv(self, filename)
Quote.__repr__(self)
googleIntradayQuote.__init__(self, symbol, interval_seconds = 300, num_days = 5)
googleQuote.__init__(self, symbol, start_date, end_date = datetime.date.today()


utilmy/zarchive/storage/panda_util.py
-------------------------functions----------------------
excel_topandas(filein, fileout)
panda_toexcel()
panda_todabatase()
database_topanda()
sqlquery_topanda()
folder_topanda()
panda_tofolder()
numpy_topanda(vv, fileout = "", colname = "data")
panda_tonumpy(filename, nsize, tablen = 'data')
df_topanda(vv, filenameh5, colname = 'data')
load_frompanda(filenameh5, colname = "data")
csv_topanda(filein1, filename, tablen = 'data', lineterminator=",")
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
excel_topanda(filein, fileout)
array_toexcel(vv, wk, r1)subset = 'rownum', take_last=True)level=0))a) = True)level=0))a):)
unique_rows(a)
remove_zeros()
sort_array()



utilmy/zarchive/storage/multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy/zarchive/storage/technical_indicator.py
-------------------------functions----------------------
np_find(item, vec)
np_find_minpos(values)
np_find_maxpos(values)
date_earningquater(t1)
date_option_expiry(date)
linearreg(a, *args)
np_sortbycolumn(arr, colid, asc = True)
np_findlocalmax(v)
findhigher(item, vec)
findlower(item, vec)
np_findlocalmin(v)
supportmaxmin1(df1)
RET(df, n)
qearning_dist(df)
optionexpiry_dist(df)
nbtime_reachtop(df, n, trigger = 0.005)
nbday_high(df, n)
distance_day(df, tk, tkname)
distance(df, tk, tkname)
MA(df, n)
EMA(df, n)
MOM(df, n)
ROC(df, n)
ATR(df, n)
BBANDS(df, n)
PPSR(df)
STOK(df)
STO(df)
TRIX(df, n)
ADX(df, n, n_ADX)
MACD(df, n_fast, n_slow)
MassI(df)
Vortex(df, n)
KST(df, r1, r2, r3, r4, n1, n2, n3, n4)
RSI(df, n = 14)
RMI(df, n = 14, m = 10)
TSI(df, r, s)
ACCDIST(df, n)
Chaikin(df)
MFI(df, n)
OBV(df, n)
FORCE(df, n)
EOM(df, n)
CCI(df, n)
COPP(df, n)
KELCH(df, n)
ULTOSC(df)
DONCH(df, n)
STDDEV(df, n)
RWI(df, nn, nATR)
nbday_low(df, n)
nbday_high(df, n)



utilmy/zarchive/storage/allmodule.py
-------------------------functions----------------------
aa_isanaconda()



utilmy/zarchive/storage/dbcheck.py


utilmy/zarchive/storage/stateprocessor.py
-------------------------functions----------------------
sort(x, col, asc)
perf(close, t0, t1)
and2(tuple1)
ff(x, symfull = symfull)
gap(close, t0, t1, lag)
process_stock(stkstr, show1 = 1)
printn(ss, symfull = symfull, s1 = s1)
show(ll, s1 = s1)
get_treeselect(stk, s1 = s1, xnewdata = None, newsample = 5, show1 = 1, nbtree = 5, depthtree = 10)
store_patternstate(tree, sym1, theme, symfull = symfull)
load_patternstate(name1)
get_stocklist(clf, s11, initial, show1 = 1)



utilmy/zarchive/storage/rec_metrics.py
-------------------------functions----------------------
predict(model, uid, pids)
precision_at_k(model, ground_truth, k, user_features = None, item_features = None)
full_auc(model, ground_truth)



utilmy/zarchive/storage/installNewPackage.py


utilmy/zarchive/storage/derivatives.py
-------------------------functions----------------------
loadbrownian(nbasset, step, nbsimul)
dN(d)
dN2d(x, y)
N(d)
d1f(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
bsbinarycall(S0, K, t, T, r, d, vol)
bscall(S0, K, t, T, r, d, vol)
bsput(S0, K, t, T, r, d, vol)
bs(S0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsgamma(St, K, t, T, r, d, vol, cp)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bstheta(St, K, t, T, r, d, vol, cp)
bsrho(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsdvd(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)
gdelta(St, K, t, T, r, d, vol, pv)
ggamma(St, K, t, T, r, d, vol, pv)
gvega(St, K, t, T, r, d, vol, pv)
gtheta(St, K, t, T, r, d, vol, pv)
genmatrix(ni, nj, gg)
gensymmatrix(ni, nj, pp)
timegrid(timestep, maturityyears)
generateall_multigbm1(process, ww, s0, mu, vol, corrmatrix, timegrid, nbsimul, nproc = -1, type1 = -1, strike = 0.0, cp = 1)
logret_to_ret(log_returns)
logret_to_price(s0, log_ret)
brownian_logret(mu, vol, timegrid)
brownian_process(s0, vol, timegrid)
gbm_logret(mu, vol, timegrid)
gbm_process(s0, mu, vol, timegrid)
gbm_process_euro(s0, mu, vol, timegrid)
gbm_process2(s0, mu, vol, timegrid)
generateallprocess(process, params01, timegrid1, nbsimul)
generateallprocess_gbmeuro(process, params01, timegrid1, nbsimul)
getpv(discount, payoff, allpriceprocess)
multigbm_processfast(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
getbrowniandata(nbasset, step, simulk)
multigbm_processfast2(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultigbmfast(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
multigbm_processfast3(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultigbmfast2(process, s0, mu, vol, corrmatrix, timegrid, nbsimul, type1)
multibrownian_logret(mu, vol, corrmatrix, timegrid)
multigbm_logret(mu, vol, corrmatrix, timegrid)
multilogret_to_price(s0, log_ret)
multigbm_process(s0, voldt, drift, upper_cholesky, nbasset, n, kk)
generateallmultiprocess(process, s0, mu, vol, corrmatrix, timegrid, nbsimul)
jump_process(lamda, jumps_mu, jumps_vol, timegrid)
gbmjump_logret(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
gbmjump_process(s0, mu, vol, lamda, jump_mu, jump_vol, timegrid)
lgnormalmoment1(ww, fft, vol, corr, tt)
lgnormalmoment2(ww, fft, vol, corr, tt)
lgnormalmoment3(ww, fft, vol, corr, tt)
lgnormalmoment4(ww, fft, vol, corr, tt)
solve_momentmatch3(ww, b0, fft, vol, corr, tt)
savebrownian(nbasset, step, nbsimul)
plot_greeks(function, greek)
plot_greeks(function, greek)
plot_values(function)
CRR_option_value(S0, K, T, r, vol, otype, M = 4)



utilmy/zarchive/storage/sobol.py
-------------------------functions----------------------
convert_csv2hd5f(filein1, filename)
getrandom_tonumpy(filename, nbdim, nbsample)
comoment(xx, yy, nsample, kx, ky)
acf(data)
getdvector(dimmax, istart, idimstart)
pathScheme_std(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_(T, n, zz)
testdensity(nsample, totdim, bin01, Ti = -1)
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
lognormal_process2d(a1, z1, a2, z2, k)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)
call_process(a, z, k)
binary_process(a, z, k)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
numexpr_vect_calc(filename, i0, imax, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )
plot_outlier(fileoutlier, kk)
permute(yy, kmax)
permute2(xx, yy, kmax)



utilmy/zarchive/storage/benchmarktest.py
-------------------------functions----------------------
payoff1(pricepath)
payoffeuro1(st)
payoff2(pricepath)
payoffeuro1(st)



utilmy/zarchive/storage/symbolicmath.py
-------------------------functions----------------------
spp()
print2(a0, a1 = '', a2 = '', a3 = '', a4 = '', a5 = '', a6 = '', a7 = '', a8 = '')
factorpoly(pp)
EEvarbrownian(ff1d)
EEvarbrownian2d(ff)
lagrangian2d(ll)
decomposecorrel(m1)
nn(x)
nn2(x, y, p)
dnn2(x, y, p)
dnn(x, y, p)
taylor2(ff, x0, n)
diffn(ff, x0, kk)
dN(x)
N(x)
d1f(St, K, t, T, r, d, vol)
d2f(St, K, t, T, r, d, vol)
d1xf(St, K, t, T, r, d, vol)
d2xf(St, K, t, T, r, d, vol)
bsbinarycall(s0, K, t, T, r, d, vol)
bscall(s0, K, t, T, r, d, vol)
bsput(s0, K, t, T, r, d, vol)
bs(s0, K, t, T, r, d, vol)
bsdelta(St, K, t, T, r, d, vol, cp1)
bsstrikedelta(s0, K, t, T, r, d, vol, cp1)
bsstrikegamma(s0, K, t, T, r, d, vol)
bsgamma(St, K, t, T, r, d, vol, cp)
bstheta(St, K, t, T, r, d, vol, cp)
bsrho(St, K, t, T, r, d, vol, cp)
bsvega(St, K, t, T, r, d, vol, cp)
bsdvd(St, K, t, T, r, d, vol, cp)
bsvanna(St, K, t, T, r, d, vol, cp)
bsvolga(St, K, t, T, r, d, vol, cp)
bsgammaspot(St, K, t, T, r, d, vol, cp)



utilmy/zarchive/storage/testmulti.py
-------------------------functions----------------------
mc01()
mc02()
serial(samples, x, widths)
multiprocess(processes, samples, x, widths)
test01()
random_tree(Data)
random_tree(Data)
test01()



utilmy/zarchive/storage/codeanalysis.py
-------------------------functions----------------------
wi(*args)
printinfile(vv, file2)
wi2(*args)
indent()
dedent()
describe_builtin(obj)
describe_func(obj, method = False)
describe_klass(obj)
describe(obj)
describe_builtin2(obj, name1)
describe_func2(obj, method = False, name1 = '')
describe_func3(obj, method = False, name1 = '')
describe_klass2(obj, name1 = '')
describe2(module, type1 = 0)
getmodule_doc(module1, file2 = '')



utilmy/zarchive/storage/dl_utils.py
-------------------------functions----------------------
save_weights(file, tuple_weights)
save_prediction(file, prediction)
log(msg, file = "")
logfile(msg, file)
log_p(msg, file = "")
init_weight(hidden1, hidden2, acti_type)
get_all_data(file)
get_batch_data(file, index, size)
get_xy(line)
file_len(fname)
feats_len(fname)



utilmy/zarchive/py2to3/_HELP.py
-------------------------functions----------------------
os_compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
os_VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)
os_VS_start(self, version)
fun_cython(a)
fun_python(a)



utilmy/zarchive/py2to3/global01.py


utilmy/zarchive/py2to3/excel.py
-------------------------functions----------------------
get_workbook_name()
double_sum(x, y)
add_one(data)
matrix_mult(x, y)
npdot()



utilmy/zarchive/py2to3/kagglegym.py
-------------------------functions----------------------
r_score(y_true, y_pred, sample_weight = None, multioutput = None)
make()

-------------------------methods----------------------
Observation.__init__(self, train, target, features)
Environment.__init__(self)
Environment.reset(self)
Environment.step(self, target)
Environment.__str__(self)


utilmy/zarchive/py2to3/fast.py
-------------------------functions----------------------
day(s)
month(s)
year(s)
hour(s)
weekday(s)
season(d)
daytime(d)
fastStrptime(val, format)
drawdown_calc_fast(price)
std(x)
mean(x)
_compute_overlaps(u, v)
distance_jaccard2(u, v)
distance_jaccard(u, v)
distance_jaccard_X(X)
cosine(u, v)
rmse(y, yhat)
cross(vec1, vec2)
norm(vec)
log_exp_sum2(a, b)



utilmy/zarchive/py2to3/allmodule_fin.py


utilmy/zarchive/py2to3/datanalysis.py
-------------------------functions----------------------
pd_filter_column(df_client_product, filter_val = [], iscol = 1)
pd_missing_show()
pd_validation_struct()
pd_checkpoint()
xl_setstyle(file1)
xl_val(ws, colj, rowi)
isnull(x)
xl_get_rowcol(ws, i0, j0, imax, jmax)
pd_stack_dflist(df_list)
xl_getschema(dirxl = "", filepattern = '*.xlsx', dirlevel = 1, outfile = '.xlsx')
str_to_unicode(x, encoding = 'utf-8')
csv_dtypes_getdict(df = None, csvfile = None)
csv_fast_processing()
csv_col_schema_toexcel(dircsv = "", filepattern = '*.csv', outfile = '.xlsx', returntable = 1, maxrow = 5000000, maxcol_pertable = 90, maxstrlen = 'U80')
csv_col_get_dict_categoryfreq(dircsv, filepattern = "*.csv", category_cols = [], maxline = -1, fileencoding = "utf-8")
csv_row_reduce_line(fromfile, tofile, condfilter, catval_tokeep, maxline = -1)
csv_analysis()
csv_row_reduce_line_manual(file_category, file_transact, file_reduced)
csv_row_mapreduce(dircsv = "", outfile = "", type_mapreduce = 'sum', chunk =  5000000)
csv_pivotable(dircsv = "", filepattern = '*.csv', fileh5 = '.h5', leftX = 'col0', topY = 'col2', centerZ = 'coli', mapreduce = 'sum', chunksize =  500000, tablename = 'df')
csv_bigcompute()
db_getdata()
db_sql()
db_meta_add(metadb, dbname, new_table = ('', [])
db_meta_find(ALLDB, query = '', filter_db = [], filter_table = [], filter_column = [])
col_study_getcategorydict_freq(catedict)
col_feature_importance(Xcol, Ytarget)
col_study_distribution_show(df, col_include = None, col_exclude = None, pars={'binsize' = {'binsize':20})
col_study_summary(Xmat = [0.0, 0.0], Xcolname = ['col1', 'col2'], Xcolselect = [9, 9], isprint = 0)
col_pair_plot(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
col_pair_correl(Xcol, Ytarget)
col_pair_interaction(Xcol, Ytarget)
plot_col_pair(dfX, Xcolname_selectlist = None, dfY = None, Ycolname = None)
tf_transform_catlabel_toint(Xmat)
tf_transform_pca(Xmat, dimpca = 2, whiten = True)
plot_distance_heatmap(Xmat_dist, Xcolname)
plot_cluster_2D(X_2dim, target_class, target_names)
plot_cluster_tsne(Xmat, Xcluster_label = None, metric = 'euclidean', perplexity = 50, ncomponent = 2, savefile = '', isprecompute = False, returnval = True)
plot_cluster_pca(Xmat, Xcluster_label = None, metric = 'euclidean', dimpca = 2, whiten = True, isprecompute = False, savefile = '', doreturn = 1)
plot_cluster_hiearchy(Xmat_dist, p = 30, truncate_mode = None, color_threshold = None, get_leaves = True, orientation = 'top', labels = None, count_sort = False, distance_sort = False, show_leaf_counts = True, no_plot = False, no_labels = False, leaf_font_size = None, leaf_rotation = None, leaf_label_func = None, show_contracted = False, link_color_func = None, ax = None, above_threshold_color = 'b')
plot_distribution_density(Xsample, kernel = 'gaussian', N = 10, bandwith = 1 / 10.0)
plot_Y(Yval, typeplot = '.b', tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY(xx, yy, zcolor = None, tsize = None, labels = None, title = '', xlabel = '', ylabel = '', zcolor_label = '', figsize = (8, 6)
plot_XY_plotly(xx, yy, towhere = 'url')
plot_XY_seaborn(X, Y, Zcolor = None)
optim_is_pareto_efficient(Xmat_cost, epsilon =  0.01, ret_boolean = 1)
sk_model_auto_tpot(Xmat, y, outfolder = 'aaserialize/', model_type = 'regressor/classifier', train_size = 0.5, generation = 1, population_size = 5, verbosity = 2)
sk_params_search_best(Xmat, Ytarget, model1, param_grid={'alpha' = {'alpha':  np.linspace(0, 1, 5) }, method = 'gridsearch', param_search= {'scoretype' =  {'scoretype':'r2', 'cv':5, 'population_size':5, 'generations_number':3 })
sk_distribution_kernel_bestbandwidth(kde)
sk_distribution_kernel_sample(kde = None, n = 1)
sk_correl_rank(correl = [[1, 0], [0, 1]])
sk_error_r2(Ypred, y_true, sample_weight = None, multioutput = None)
sk_error_rmse(Ypred, Ytrue)
sk_cluster_distance_pair(Xmat, metric = 'jaccard')
sk_cluster(Xmat, metric = 'jaccard')
sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval = 1)
sk_optim_de(obj_fun, bounds, maxiter = 1, name1 = '', solver1 = None, isreset = 1, popsize = 15)
sk_feature_importance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, isprint1 = 1, njobs = 1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")

-------------------------methods----------------------
sk_model_template1.__init__(self, alpha = 0.5, low_y_cut = -0.09, high_y_cut = 0.09, ww0 = 0.95)
sk_model_template1.fit(self, X, Y = None)
sk_model_template1.predict(self, X, y = None, ymedian = None)
sk_model_template1.score(self, X, Ytrue = None, ymedian = None)
sk_stateRule.__init__(self, state, trigger, colname = [])
sk_stateRule.addrule(self, rulefun, name = '', desc = '')
sk_stateRule.eval(self, idrule, t, ktrig = 0)
sk_stateRule.help()


utilmy/zarchive/py2to3/alldata.py


utilmy/zarchive/py2to3/report.py
-------------------------functions----------------------
map_show()
xl_create_pivot(infile, index_list = ["Manager", "Rep", "Product"], value_list = ["Price", "Quantity"])
xl_save_report(report, outfile)
xl_create_pdf()



utilmy/zarchive/py2to3/linux.py
-------------------------functions----------------------
load_session(name = 'test_20160815')
save_session(name = '')
isfloat(value)
isint(x)
aa_isanaconda()
aa_cleanmemory()
aa_getmodule_doc(module1, fileout = '')
np_interpolate_nan(y)
and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
sortcol(arr, colid, asc = 1)
sort(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_findfirst(item, vec)
np_find(item, vec)
find(item, vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = True)
np_memory_array_adress(x)
sk_featureimportance(clfrf, feature_name)
sk_showconfusion(clfrf, X_train, Y_train, isprint = True)
sk_tree(Xtrain, Ytrain, nbtree, maxdepth, print1)
sk_gen_ensemble_weight(vv, acclevel, maxlevel = 0.88)
sk_votingpredict(estimators, voting, ww, X_test)
sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base = " ")
pd_array_todataframe(price, symbols = None, date1 = None, dotranspose = False)
pd_date_intersection(qlist)
pd_resetindex(df)
pd_create_colmap_nametoid(df)
pd_dataframe_toarray(df)
pd_changeencoding(data, cols)
pd_createdf(val1, col1 = None, idx1 = None)
pd_insertcolumn(df, colname, vec)
pd_insertrows(df, rowval, index1 = None)
pd_replacevalues(df, matrix)
pd_storeadddf(df, dfname, dbfile='F = 'F:\temp_pandas.h5')
pd_storedumpinfo(dbfile='E = 'E:\_data\stock\intraday_google.h5')
pd_remove_row(df, row_list_index = [23, 45])
pd_extract_col_idx_val(df)
pd_split_col_idx_val(df)
pd_addcolumn(df1, name1 = 'new')
pd_removecolumn(df1, name1)
pd_save_vectopanda(vv, filenameh5)
pd_load_panda2vec(filenameh5, store_id = 'data')
pd_csv_topanda(filein1, filename, tablen = 'data')
pd_getpanda_tonumpy(filename, nsize, tablen = 'data')
pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
sk_cluster_kmeans(x, nbcluster = 5, isplot = True)
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_toint(datelist1)
dateint_todatetime(datelist1)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_as_float(dt)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
datetime_tostring(datelist1)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datestring_toint(datelist1)
numexpr_vect_calc(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, expr, i0 = 0, imax = 1000, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
textvect_topanda(vv, fileout = "")
comoment(xx, yy, nsample, kx, ky)
acf(data)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
compileVSsolution(dir1, flags1 = "", type1 = "devenv", compilerdir = "")
VS_start(self, version)
VS_build(self, lib_to_build)
set_rc_version(rcfile, target_version)



utilmy/zarchive/py2to3/portfolio.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
date_finddateid(date1, dateref)
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy/zarchive/py2to3/multiprocessfunc.py
-------------------------functions----------------------
multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky, nbasset, n, price, type1 = 0, strike = 0, cp = 1)
func(val, lock)
multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky, nbasset, n, price)
bm_generator(bm, dt, n, type1)
merge(d2)
integratenp2(its, nchunk)
integratenp(its, nchunk)
integratene(its)
parzen_estimation(x_samples, point_x, h)
init2(d)
init_global1(l, r)
np_sin(value)
ne_sin(x)
res_shared2()
list_append(count, id, out_list)



utilmy/zarchive/py2to3/fast_parallel.py
-------------------------functions----------------------
task_summary(tasks)
task_progress(tasks)
task_find_best(tasks, n_top = 5)
task_parallel_job_01(name, param, datadict)



utilmy/zarchive/py2to3/util_min.py
-------------------------functions----------------------
os_wait_cpu(priority = 300, cpu_min = 50)
isexist(a)
isfloat(x)
isint(x)
a_isanaconda()
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = '/zdisks3/output', zipname = '/zdisk3/output.zip', dir_prefix = None, iscompress = True)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = 'zdisk/test', isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = '', to_folder = '', my_log='H = 'H:/robocopy_log.txt')
os_file_replacestring1(findStr, repStr, filePath)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_print_tofile(vv, file1, mode1 = 'a')
a_get_pythonversion()
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_split_dir_file(dirfile)
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = '/folder1/keyname', isabsolutpath = 0)
load(folder = '/folder1/keyname', isabsolutpath = 0)
save_test(folder = '/folder1/keyname', isabsolutpath = 0)
py_save_obj(obj, folder = '/folder1/keyname', isabsolutpath = 0)
py_load_obj(folder = '/folder1/keyname', isabsolutpath = 0, encoding1 = 'utf-8')
z_key_splitinto_dir_name(keyname)



utilmy/zarchive/py2to3/function_custom.py
-------------------------functions----------------------
mapping_calc_risk_elvis_v03(ss, tr, t, riskout)
mapping_calc_risk_v02(ss, tr, t, risk0)
mapping_calc_risk_v01(ss, tr, t, risk0)
mapping_risk_ww_v01(risk, wwmat, ww2)
mapping_calc_risk_v00(self, ss, tr, t, risk0)
getweight(ww, size = (9, 3)
fun_obj(vv, ext)



utilmy/zarchive/py2to3/__init__.py


utilmy/zarchive/py2to3/coke_functions.py
-------------------------functions----------------------
date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH = 'YYYY-MM-DD HH:mm:SS')
date_diffstart(t)
date_diffend(t)
np_dict_tolist(dd)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
day(s)
month(s)
year(s)
hour(s)
weekday(s, fmt = 'YYYY-MM-DD', i0 = 0, i1 = 10)
season(d)
daytime(d)
pd_date_splitall(df, coldate = 'purchased_at')



utilmy/zarchive/py2to3/allmodule.py
-------------------------functions----------------------
pprint(table1, tablefmt = "simple")
pprint2(x)
str_convert_beforeprint(x)
str_to_utf8(x)
str_to_unicode(x, encoding = 'utf-8')



utilmy/zarchive/py2to3/geospatial.py


utilmy/zarchive/py2to3/portfolio_withdate.py
-------------------------functions----------------------
data_jpsector()
date_earningquater(t1)
date_is_3rdfriday(s)
date_option_expiry(date)
date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate)
date_find_kintraday_fromdate(d1, intradaydate1, h1 = 9, m1 = 30)
date_find_intradateid(datetimelist, stringdate = ['20160420223000'])
datetime_convertzone1_tozone2(tt, fromzone = 'Japan', tozone = 'US/Eastern')
date_extract_dailyopenclosetime(spdateref1, market = 'us')
datetime_tostring(tt)
datestring_todatetime(datelist1, format1 =  "%Y%m%d")
datetime_todate(tt)
datetime_toint(datelist1)
datetime_tointhour(datelist1)
dateint_tostring(datelist1, format1 = '%b-%y')
dateint_todatetime(datelist1)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpypdate(t, islocaltime = True)
date_diffindays(intdate1, intdate2)
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_as_float(dt)
date_todatetime(tlist)
date_removetimezone(datelist)
datediff_inyear(startdate, enddate)
date_generatedatetime(start = "20100101", nbday = 10, end = "")
date_add_bdays(from_date, add_days)
date_getspecificdate(datelist, datetype1 = "yearend", outputype1 = "intdate", includelastdate = True, includefirstdate = False, )
date_alignfromdateref(array1, dateref)
_date_align(dateref, datei, tmax, closei)
date_align(array1, dateref)
min_withposition(values)
max_withposition(values)
_reshape(x)
_notnone(x)
plot_price(asset, y2 = None, y3 = None, y4 = None, y5 = None, sym = None, savename1 = '', tickperday = 20, date1 = None, graphsize = (10, 5)
plot_priceintraday(data)
plot_check(close, tt0i = 20140102, tt1i = 20160815, dateref = [], sym = [], tickperday = 120)
plot_pricedate(date1, sym1, asset1, sym2 = None, bsk1 = None, verticaldate = None, savename1 = '', graphsize = (10, 5)
generate_sepvertical(asset1, tt, tmax, start = None, datebar = None)
save_asset_tofile(file1, asset1, asset2 = None, asset3 = None, date1 = None, title1 = None)
load_asset_fromfile(file1)
array_todataframe(price, symbols = None, date1 = None)
pd_dataframe_toarray(df)
isfloat(value)
isint(x)
correlation_mat(Xmat, type1 = "robust", type2 = "correl")
correl_reducebytrigger(correl2, trigger)
sk_cov_fromcorrel(correl, ret_close1)
cointegration(x, y)
causality_y1_y2(price2, price1, maxlag)
reg_slope(close, dateref, tlag, type1 = 'elasticv')
rolling_cointegration(x, y)
regression(yreturn, xreturn, type1 = "elasticv")
regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly = True)
regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag = 1)
regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac, priceriskfac, nlaglist)
getdiff_fromquotes(close, timelag)
getret_fromquotes(close, timelag = 1)
getlogret_fromquotes(close, timelag = 1)
getprice_fromret(ret, normprice = 100)
price_normalize100(ret, normprice = 100)
price_normalize_1d(ret, normprice = 100, dtype1 =  np.float32)
norm_fast(y, ny)
correl_fast(xn, y, nx)
volhisto_fromret(retbsk, t, volrange, axis = 0)
volhisto_fromprice(price, t, volrange, axis = 0)
volhistorolling_fromprice(price, volrange)
rsk_calc_all_TA(df = 'panda_dataframe')
ta_highbandtrend1(close2, type1 = 0)
pd_transform_asset(q0, q1, type1 = "spread")
calcbasket_table(wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000)
folio_createvolta_asset(close, vol = 0.12, volrange = 120, lev = 1.0)
folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa = 0.02, showgraph = True)
folio_leverageetf(price, lev = 1.0, costpa = 0.0)
folio_inverseetf(price, costpa = 0.0)
folio_longshort_unit(long1, short1, ww = [1, -1], costpa = 0.0, tlag = 1, istable = 1, wwschedule = [])
folio_longshort_unitfixed(long1, short1, nn = [1, -1], costpa = 0.0, tlag = 1, istable = 1)
folio_longshort_pct(long1, short1, ww = [1, -1], costpa = 0.0)
folio_histogram(close)
folio_voltarget(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta(bsk, targetvol = 0.11, volrange =  90, expocap = 1.5)
folio_volta2(bsk, riskind, par, targetvol = 0.11, volrange =  90, cap = 1.5, floor = 0.0, costbp = 0.0005)
folio_fixedunitprice(price, fixedww, costpa = 0.0)
folio_fixedweightprice(price, fixedww, costpa = 0.0)
folio_fixedweightret(ret, fixedww)
folio_cost_turnover(wwall, bsk, dateref, costbp)
folio_riskpa(ret, targetvol = 0.1, volrange = 90, cap = 1.0)
folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod = "1monthend")
folio_concenfactor2(ww, masset = 12)
calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0, nbrange, criteria)
calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs)
similarity_correl(ret_close2, funargs)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
np_similarity(x, y, wwerr = [], type1 = 0)
np_distance_l1(x, y, wwerr)
get(close, timelag)

-------------------------methods----------------------
index.__init__(self, id1, sym, ww, tstart)
index.close(self)
index.updatehisto(self)
index.help(self)
index.__init__(self, id1, sym, ww, tstart)
index.calc_baskettable_pct(self, type1 = "table", showdetail = 0)
index._wwpct_rebal(self, wwpct_actual, t, trebal)
index._udpate_wwindpct(self, t, bskt, hedgecost, wwpct_actual, wwpct_th)
index.calc_baskettable_unit()
folioCalc.__init__(self, sym, close, dateref)
folioCalc.set_symclose(self, sym, close, dateref)
folioCalc.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioCalc._weightcalc_generic(self, wwvec, t)
folioCalc._weightcalc_regime(self, wwvec, wwextra, t)
folioCalc._regimecalc(self, t, wwextra)
folioCalc._weightcalc_constant(self, ww2, t)
folioCalc.getweight(self)
folioCalc.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps = 0.000, showdetail = 0)
folioCalc.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioCalc.multiperiod_ww(self, t)
folioCalc.help(self)
folioRiskIndicator.__init__(self, sym, close, dateref)
folioRiskIndicator.set_symclose(self, sym, close, dateref)
folioRiskIndicator.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioRiskIndicator.calcrisk(self, wwvec = [], initval = 1)
folioRiskIndicator._weightcalc_generic(self, wwvec, t)
folioRiskIndicator._weightcalc_regime(self, wwvec, wwextra, t)
folioRiskIndicator._regimecalc(self, t, wwextra)
folioRiskIndicator.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.__init__(self, sym, close, dateref)
folioOptimizationF.set_symclose(self, sym, close, dateref)
folioOptimizationF.setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid = "spprice", lfun = None)
folioOptimizationF.calcbasket_obj2(self, wwvec)
folioOptimizationF.calcbasket_obj(self, wwvec)
folioOptimizationF._loss_obj(self, ww2, wwpenalty)
folioOptimizationF._weightcalc_generic(self, wwvec, t)
folioOptimizationF._weightcalc_regime(self, wwvec, wwextra, t)
folioOptimizationF._regimecalc(self, t, wwextra)
folioOptimizationF._weightcalc_constant(self, ww2, t)
folioOptimizationF.calc_optimal_weight(self, maxiter = 1, name1 = '', isreset = 1, popsize = 15)
folioOptimizationF.getweight(self)
folioOptimizationF.calc_baskettable(self, wwvec, ret, type1 = "table", wwtype = "constant", rebfreq = 1, costbps =  0.000, showdetail = 0)
folioOptimizationF.plot(self, wwvec = None, show1 = 1, tickperday = 60)
folioOptimizationF._objective_criteria(self, bsk)
folioOptimizationF.multiperiod_ww(self, t)
folioOptimizationF.help(self)
folioOptimizationF.mapping_risk_ww(self, risk, wwmat, ww2 = self.wwasset0)
folioOptimizationF._mapping_calc_risk(self, ss, tr, t, risk0)
searchSimilarity.__init__(self, filejpstock=r'E = r'E:/_data/stock/daily/20160616/jp', sym01 = ['7203'], symname = ['Toyota'], startdate =  20150101, enddate = 20160601, pricetype = "close")
searchSimilarity.load_quotes_fromdb(self, picklefile = '')
searchSimilarity.__generate_return__(self, nlag)
searchSimilarity.__overweight__(self, px)
searchSimilarity.set_searchcriteria(self, name1 = '7203', date1 = 20160301, date2 = 20160601, nlag = 1, searchperiodstart = 20120101, typesearch = "pattern2", )
searchSimilarity.launch_search(self)
searchSimilarity.show_comparison_graph(self, maxresult = 20, show_only_different_time = True, fromid = 0, fromend =  0, filenameout = '')
searchSimilarity.staticmethod(self, x)
searchSimilarity.get_rankresult(self, filetosave = '')
searchSimilarity.export_results()


utilmy/zarchive/py2to3/filelock.py
-------------------------methods----------------------
FileLock.__init__(self, protected_file_path, timeout = None, delay = 1, lock_file_contents = None)
FileLock.locked(self)
FileLock.available(self)
FileLock.acquire(self, blocking = True)
FileLock.release(self)
FileLock.__enter__(self)
FileLock.__exit__(self, type, value, traceback)
FileLock.__del__(self)
FileLock.purge(self)


utilmy/zarchive/py2to3/util_ml.py
-------------------------functions----------------------
create_weight_variable(name, shape)
create_bias_variable(name, shape)
create_adam_optimizer(learning_rate, momentum)
tf_check()
parse_args(ppa = None, args =  {})
parse_args2(ppa = None)
tf_global_variables_initializer(sess = None)
visualize_result()

-------------------------methods----------------------
TextLoader.__init__(self, data_dir, batch_size, seq_length)
TextLoader.preprocess(self, input_file, vocab_file, tensor_file)
TextLoader.load_preprocessed(self, vocab_file, tensor_file)
TextLoader.create_batches(self)
TextLoader.next_batch(self)
TextLoader.reset_batch_pointer(self)


utilmy/zarchive/py2to3/utilgeo.py
-------------------------functions----------------------
df_to_geojson(df, col_properties, lat = 'latitude', lon = 'longitude')



utilmy/zarchive/py2to3/rstatpy.py
-------------------------functions----------------------
stl(data, ns, np = None, nt = None, nl = None, isdeg = 0, itdeg = 1, ildeg = 1, nsjump = None, ntjump = None, nljump = None, ni = 2, no = 0, fulloutput = False)



utilmy/zarchive/zzarchive/zutil.py
-------------------------functions----------------------
session_load_function(name = "test_20160815")
session_save_function(name = "test")
py_save_obj_dill(obj1, keyname = "", otherfolder = 0)
aa_unicode_ascii_utf8_issue()
isfloat(x)
isint(x)
isanaconda()
a_run_ipython(cmd1)
py_autoreload()
os_platform()
a_start_log(id1 = "", folder = "aaserialize/log/")
a_cleanmemory()
a_info_conda_jupyter()
a_run_cmd(cmd1)
a_help()
print_object_tofile(vv, txt, file1="d = "d:/regression_output.py")
print_progressbar(iteration, total, prefix = "", suffix = "", decimals = 1, bar_length = 100)
os_zip_checkintegrity(filezip1)
os_zipfile(folderin, folderzipname, iscompress = True)
os_zipfolder(dir_tozip = "/zdisks3/output", zipname = "/zdisk3/output.zip", dir_prefix = True, iscompress=Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[ = Trueimport shutil_ = iscompressdir_tozip = dir_tozip if dir_tozip[-1] != "/" else dir_tozip[:-1]if dir_prefix:)
os_zipextractall(filezip_or_dir = "folder1/*.zip", tofolderextract = "zdisk/test", isprint = 1)
os_folder_copy(src, dst, symlinks = False, pattern1 = "*.py", fun_file_toignore = None)
os_folder_create(directory)
os_folder_robocopy(from_folder = "", to_folder = "", my_log="H = "H:/robocopy_log.txt")
os_file_replace(source_file_path, pattern, substring)
os_file_replacestring1(find_str, rep_str, file_path)
os_file_replacestring2(findstr, replacestr, some_dir, pattern = "*.*", dirlevel = 1)
os_file_getname(path)
os_file_getpath(path)
os_file_gettext(file1)
os_file_listall(dir1, pattern = "*.*", dirlevel = 1, onlyfolder = 0)
_os_file_search_fast(fname, texts = None, mode = "regex/str")
os_file_search_content(srch_pattern = None, mode = "str", dir1 = "", file_pattern = "*.*", dirlevel = 1)
os_file_rename(some_dir, pattern = "*.*", pattern2 = "", dirlevel = 1)
os_gui_popup_show(txt)
os_print_tofile(vv, file1, mode1 = "a")
os_path_norm(pth)
os_path_change(path1)
os_path_current()
os_file_exist(file1)
os_file_size(file1)
os_file_read(file1)
os_file_isame(file1, file2)
os_file_get_extension(file_path)
os_file_normpath(path)
os_folder_is_path(path_or_stream)
os_file_get_path_from_stream(maybe_stream)
os_file_try_to_get_extension(path_or_strm)
os_file_are_same_file_types(paths)
os_file_norm_paths(paths, marker = "*")
os_file_mergeall(nfile, dir1, pattern1, deepness = 2)
os_file_extracttext(output_file, dir1, pattern1 = "*.html", htmltag = "p", deepness = 2)
os_path_append(p1, p2 = None, p3 = None, p4 = None)
os_wait_cpu(priority = 300, cpu_min = 50)
os_split_dir_file(dirfile)
os_process_run(cmd_list, capture_output = False)
os_process_2()
py_importfromfile(modulename, dir1)
py_memorysize(o, ids, hint = " deep_getsizeof(df_pd, set()
save(obj, folder = "/folder1/keyname", isabsolutpath = 0)
load(folder = "/folder1/keyname", isabsolutpath = 0)
save_test(folder = "/folder1/keyname", isabsolutpath = 0)
py_save_obj(obj1, keyname = "", otherfolder = 0)
py_load_obj(folder = "/folder1/keyname", isabsolutpath = 0, encoding1 = "utf-8")
z_key_splitinto_dir_name(keyname)
os_config_setfile(dict_params, outfile, mode1 = "w+")
os_config_getfile(file1)
os_csv_process(file1)
pd_toexcel(df, outfile = "file.xlsx", sheet_name = "sheet1", append = 1, returnfile = 1)
pd_toexcel_many(outfile = "file1.xlsx", df1 = None, df2 = None, df3 = None, df4 = None, df5 = None, df6 = Nonedf1, outfile, sheet_name="df1")if df2 is not None = "df1")if df2 is not None:)
find_fuzzy(xstring, list_string)
str_match_fuzzy(xstring, list_string)
str_parse_stringcalendar(cal)
str_make_unicode(input_str, errors = "replace")
str_empty_string_array(x, y = 1)
str_empty_string_array_numpy(nx, ny = 1)
str_isfloat(value)
str_is_azchar(x)
str_is_az09char(x)
str_reindent(s, num_spaces)
str_split2(delimiters, string, maxsplit = 0)
str_split_pattern(sep2, ll, maxsplit = 0)
pd_str_isascii(x)
str_to_utf8(x)
str_to_unicode(x, encoding = "utf-8")
np_minimize(fun_obj, x0 = None, argext = (0, 0)
np_minimize_de(fun_obj, bounds, name1, maxiter = 10, popsize = 5, solver = None)
np_remove_na_inf_2d(x)
np_addcolumn(arr, nbcol)
np_addrow(arr, nbrow)
np_int_tostr(i)
np_dictordered_create()
np_list_unique(seq)
np_list_tofreqdict(l1, wweight = None)
np_list_flatten(seq)
np_dict_tolist(dd, withkey = 0)
np_dict_tostr_val(dd)
np_dict_tostr_key(dd)
np_removelist(x0, xremove = None)
np_transform2d_int_1d(m2d, onlyhalf = False)
np_mergelist(x0, x1)
np_enumerate2(vec_1d)
np_pivottable_count(mylist)
np_nan_helper(y)
np_interpolate_nan(y)
np_and1(x, y, x3 = None, x4 = None, x5 = None, x6 = None, x7 = None, x8 = None)
np_sortcol(arr, colid, asc = 1)
np_ma(vv, n)
np_cleanmatrix(m)
np_torecarray(arr, colname)
np_sortbycolumn(arr, colid, asc = True)
np_sortbycol(arr, colid, asc = True)
np_min_kpos(arr, kth)
np_max_kpos(arr, kth)
np_findfirst(item, vec)
np_find(item, vec)
find(xstring, list_string)
findnone(vec)
findx(item, vec)
finds(itemlist, vec)
findhigher(x, vec)
findlower(x, vec)
np_find_minpos(values)
np_find_maxpos(values)
np_find_maxpos_2nd(numbers)
np_findlocalmax2(v, trig)
np_findlocalmin2(v, trig)
np_findlocalmax(v, trig)
np_findlocalmin(v, trig)
np_stack(v1, v2 = None, v3 = None, v4 = None, v5 = None)
np_uniquerows(a)
np_remove_zeros(vv, axis1 = 1)
np_sort(arr, colid, asc = 1)
np_memory_array_adress(x)
np_pivotable_create(table, left, top, value)
pd_info(df, doreturn = 1)
pd_info_memsize(df, memusage = 0)
pd_row_findlast(df, colid = 0, emptyrowid = None)
pd_row_select(df, **conditions)
pd_csv_randomread(filename, nsample = 10000, filemaxline = -1, dtype = None)
pd_array_todataframe(array, colname = None, index1 = None, dotranspose = False)
pd_dataframe_toarray(df)
pd_createdf(array1, col1 = None, idx1 = None)
pd_create_colmapdict_nametoint(df)
pd_extract_col_idx_val(df)
pd_extract_col_uniquevalue_tocsv(df, colname = "", csvfile = "")
pd_split_col_idx_val(df)
pd_splitdf_inlist(df, colid, type1 = "dict")
pd_find(df, regex_pattern = "*", col_restrict = None, isnumeric = False, doreturnposition = False)
pd_dtypes_totype2(df, columns = ()
pd_dtypes(df, columns = ()
pd_df_todict2(df, colkey = "table", excludekey = ("", )
pd_df_todict(df, colkey = "table", excludekey = ("", )
pd_col_addfrom_dfmap(df, dfmap, colkey, colval, df_colused, df_colnew, exceptval = -1, inplace = Truedfmap, colkey = colkey, colval=colval)rowi) = colval)rowi):)
pd_applyfun_col(df, newcol, ff, use_colname = "all/[colname]")
pd_date_intersection(qlist)
pd_is_categorical(z)
pd_str_encoding_change(df, cols, fromenc = "iso-8859-1", toenc = "utf-8")
pd_str_unicode_tostr(df, targetype = str)
pd_dtypes_type1_totype2(df, fromtype = str, targetype = str)
pd_resetindex(df)
pd_insertdatecol(df, col, format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
pd_replacevalues(df, matrix)
pd_removerow(df, row_list_index = (23, 45)
pd_removecol(df1, name1)
pd_insertrow(df, rowval, index1 = None, isreset = 1)
pd_h5_cleanbeforesave(df)
pd_h5_addtable(df, tablename, dbfile="F = "F:\temp_pandas.h5")
pd_h5_tableinfo(filenameh5, table)
pd_h5_dumpinfo(dbfile=r"E = r"E:\_data\stock\intraday_google.h5")
pd_h5_save(df, filenameh5="E = "E:/_data/_data_outlier.h5", key = "data")
pd_h5_load(filenameh5="E = "E:/_data/_data_outlier.h5", table_id = "data", exportype = "pandas", rowstart = -1, rowend = -1, ), )
pd_h5_fromcsv_tohdfs(dircsv = "dir1/dir2/", filepattern = "*.csv", tofilehdfs = "file1.h5", tablename = "df", ), dtype0 = None, encoding = "utf-8", chunksize = 2000000, mode = "a", form = "table", complib = None, )
pd_np_toh5file(numpyarr, fileout = "file.h5", table1 = "data")
date_allinfo()
datetime_tostring(datelist1)
date_remove_bdays(from_date, add_days)
date_add_bdays(from_date, add_days)
datenumpy_todatetime(tt, islocaltime = True)
datetime_tonumpydate(t, islocaltime = True)
datestring_todatetime(datelist1, format1 = "%Y%m%d")
datetime_toint(datelist1)
date_holiday()
date_add_bday(from_date, add_days)
dateint_todatetime(datelist1)
date_diffinday(intdate1, intdate2)
date_diffinbday(intd2, intd1)
date_gencalendar(start = "2010-01-01", end = "2010-01-15", country = "us")
date_finddateid(date1, dateref)
datestring_toint(datelist1)
date_now(i = 0)
date_nowtime(type1 = "str", format1="%Y-%m-%d %H = "%Y-%m-%d %H:%M:%S:%f")
date_generatedatetime(start = "20100101", nbday = 10, end = "")
np_numexpr_vec_calc()



utilmy/zarchive/zzarchive/zutil_features.py
-------------------------functions----------------------
log(*s, n = 0, m = 1, **kw)
log2(*s, **kw)
log3(*s, **kw)
os_get_function_name()
os_getcwd()
pa_read_file(path =   'folder_parquet/', cols = None, n_rows = 1000, file_start = 0, file_end = 100000, verbose = 1, )
pa_write_file(df, path =   'folder_parquet/', cols = None, n_rows = 1000, partition_cols = None, overwrite = True, verbose = 1, filesystem  =  'hdfs')
test_get_classification_data(name = None)
params_check(pars, check_list, name = "")
save_features(df, name, path = None)
load_features(name, path)
save_list(path, name_list, glob)
save(df, name, path = None)
load(name, path)
pd_read_file(path_glob = "*.pkl", ignore_index = True, cols = None, verbose = False, nrows = -1, concat_sort = True, n_pool = 1, drop_duplicates = None, col_filter = None, col_filter_val = None, **kw)
load_dataset(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_spark_koalas(path_data_x, path_data_y = '', colid = "jobId", n_sample = -1)
fetch_dataset(url_dataset, path_target = None, file_target = None)
load_function_uri(uri_name="myfolder/myfile.py = "myfolder/myfile.py::myFunction")
metrics_eval(metric_list = ["mean_squared_error"], ytrue = None, ypred = None, ypred_proba = None, return_dict = False)
pd_stat_dataset_shift(dftrain, dftest, colused, nsample = 10000, buckets = 5, axis = 0)
pd_stat_datashift_psi(expected, actual, buckettype = 'bins', buckets = 10, axis = 0)
feature_importance_perm(clf, Xtrain, ytrain, cols, n_repeats = 8, scoring = 'neg_root_mean_squared_error', show_graph = 1)
feature_selection_multicolinear(df, threshold = 1.0)
feature_correlation_cat(df, colused)
pd_feature_generate_cross(df, cols, cols_cross_input = None, pct_threshold = 0.2, m_combination = 2)
pd_col_to_onehot(dfref, colname = None, colonehot = None, return_val = "dataframe,column")
pd_colcat_mergecol(df, col_list, x0, colid = "easy_id")
pd_colcat_tonum(df, colcat = "all", drop_single_label = False, drop_fact_dict = True)
pd_colcat_mapping(df, colname)
pd_colcat_toint(dfref, colname, colcat_map = None, suffix = None)
pd_colnum_tocat(df, colname = None, colexclude = None, colbinmap = None, bins = 5, suffix = "_bin", method = "uniform", na_value = -1, return_val = "dataframe,param", params={"KMeans_n_clusters" = {"KMeans_n_clusters": 8, "KMeans_init": 'k-means++', "KMeans_n_init": 10,"KMeans_max_iter": 300, "KMeans_tol": 0.0001, "KMeans_precompute_distances": 'auto',"KMeans_verbose": 0, "KMeans_random_state": None,"KMeans_copy_x": True, "KMeans_n_jobs": None, "KMeans_algorithm": 'auto'})
pd_colnum_normalize(df0, colname, pars, suffix = "_norm", return_val = 'dataframe,param')
pd_col_merge_onehot(df, colname)
pd_col_to_num(df, colname = None, default = np.nan)
pd_col_filter(df, filter_val = None, iscol = 1)
pd_col_fillna(dfref, colname = None, method = "frequent", value = None, colgroupby = None, return_val = "dataframe,param", )
pd_pipeline_apply(df, pipeline)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
col_extractname(col_onehot)
col_remove(cols, colsremove, mode = "exact")
pd_colnum_tocat_stat(df, feature, target_col, bins, cuts = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
np_conv_to_one_col(np_array, sep_char = "_")

-------------------------methods----------------------
dict2.__init__(self, d)


utilmy/spark/src/functions/GetFamiliesFromUserAgent.py
-------------------------functions----------------------
getall_families_from_useragent(ua_string)



utilmy/spark/src/tables/table_predict_volume.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml')
preprocess(spark, conf, check = True)
model_train(df:object, conf_model:dict, verbose:bool = True)
model_predict(df:pd.DataFrame, conf_model:dict, verbose:bool = True)



utilmy/spark/src/tables/table_user_session_stats.py
-------------------------functions----------------------
run(spark:SparkSession, config_name: str = 'config.yaml')



utilmy/spark/src/tables/table_user_session_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name = 'config.yaml')



utilmy/spark/src/tables/table_predict_url_unique.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')
preprocess(spark, conf, check = True)



utilmy/spark/src/tables/table_user_log.py
-------------------------functions----------------------
run(spark:SparkSession, config_name:str)
create_userid(userlogDF:pyspark.sql.DataFrame)



utilmy/spark/src/tables/table_predict_session_length.py
-------------------------functions----------------------
run(spark:SparkSession, config_path: str = 'config.yaml', mode:str = 'train,pred')
preprocess(spark, conf, check = True)



utilmy/templates/templist/pypi_package/run_pipy.py
-------------------------functions----------------------
get_current_githash()
update_version(path, n = 1)
git_commit(message)
ask(question, ans = 'yes')
pypi_upload()
main(*args)

-------------------------methods----------------------
Version.__init__(self, major, minor, patch)
Version.__str__(self)
Version.__repr__(self)
Version.stringify(self)
Version.new_version(self, orig)
Version.parse(cls, string)


utilmy/templates/templist/pypi_package/setup.py
-------------------------functions----------------------
get_current_githash()



utilmy/zarchive/storage/aapackage_gen/global01.py


utilmy/zarchive/storage/aapackage_gen/util.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/codeanalysis.py
-------------------------functions----------------------
wi(*args)
printinfile(vv, file1)
wi2(*args)
indent()
dedent()
describe_builtin(obj)
describe_func(obj, method = False)
describe_klass(obj)
describe(obj)
describe_builtin2(obj, name1)
describe_func2(obj, method = False, name1 = '')
describe_klass2(obj, name1 = '')
describe2(module)
getmodule_doc(module1, file1 = 'moduledoc.txt')



utilmy/zarchive/storage/aapackagedev/random.py
-------------------------functions----------------------
convert_csv2hd5f(filein1, filename)
getrandom_tonumpy(filename, nbdim, nbsample)
comoment(xx, yy, nsample, kx, ky)
acf(data)
getdvector(dimmax, istart, idimstart)
pathScheme_std(T, n, zz)
pathScheme_bb(T, n, zz)
pathScheme_(T, n, zz)
testdensity(nsample, totdim, bin01, Ti = -1)
plotdensity(nsample, totdim, bin01, tit0, Ti = -1)
testdensity2d(nsample, totdim, bin01, nbasset)
lognormal_process2d(a1, z1, a2, z2, k)
testdensity2d2(nsample, totdim, bin01, nbasset, process01 = lognormal_process2d, a1 = 0.25, a2 = 0.25, kk = 1)
call_process(a, z, k)
binary_process(a, z, k)
pricing01(totdim, nsample, a, strike, process01, aa = 0.25, itmax = -1, tt = 10)
plotdensity2(nsample, totdim, bin01, tit0, process01, vol = 0.25, tt = 5, Ti = -1)
Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph)
Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep, samplejump, nsamplegraph, )
getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1 = 0.28, fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5')
outlier_clean(vv2)
overwrite_data(fileoutlier, vv2)
doublecheck_outlier(fileoutlier, ijump, nsample = 4000, trigger1 = 0.1, )fileoutlier=   'E =    'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'fileoutlier, 'data')    #from filevv5 =  pdf.values   #to numpy vectordel pdfistartx= 0; istarty= 0nsample= 4000trigger1=  0.1crrmax = 250000kk=0(crrmax, 4), dtype = 'int')  #empty listvv5)[0]0, kkmax1, 1) :  #Decrasing: dimy0 to dimmindimx =  vv5[kk, 0];   dimy =  vv5[kk, 1]y0= dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0 =  dimy * ijump + istartyym= dimy* ijump + nsample + istartyyyu1= yy1[y0:ym];   yyu2= yy2[y0:ym];   yyu3= yy3[y0:ym]x0= dimx * ijump + istartxxm= dimx* ijump + nsample + istartxxxu1= yy1[x0:xm];   xxu2= yy2[x0:xm];   xxu3= yy3[x0:xm]"sum( xxu3 * yyu1)") / (nsample) # X3.Y moments"sum( xxu1 * yyu3)") / (nsample)"sum( xxu2 * yyu2)") / (nsample)abs(c22) > trigger1)  :)
plot_outlier(fileoutlier, kk)fileoutlier, 'data')    #from filevv =  df.values   #to numpy vectordel dfxx= vv[kk, 0]yy =  vv[kk, 1]xx, yy, s = 1 )[00, 1000, 00, 1000])nsample)+'sampl D_'+str(dimx)+' X D_'+str(dimy)tit1)'_img/'+tit1+'_outlier.jpg', dpi = 100))yy, kmax))
permute(yy, kmax)
permute2(xx, yy, kmax)



utilmy/templates/templist/pypi_package/mygenerator/validate.py
-------------------------functions----------------------
image_padding_validate(final_image, min_padding, max_padding)
image_padding_load(img_path, threshold = 15)
image_padding_get(img, threshold = 0, inverse = True)
run_image_padding_validate(min_spacing: int  =  1, max_spacing: int  =  1, image_width: int  =  5, input_path: str  =  "", inverse_image: bool  =  True, config_file: str  =  "default", **kwargs, )



utilmy/templates/templist/pypi_package/mygenerator/util_exceptions.py
-------------------------functions----------------------
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy/templates/templist/pypi_package/mygenerator/dataset.py
-------------------------functions----------------------
dataset_build_meta_mnist(path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )

-------------------------methods----------------------
NlpDataset.__init__(self, meta: pd.DataFrame)
NlpDataset.__len__(self)
NlpDataset.get_sample(self, idx: int)
NlpDataset.get_text_only(self, idx: int)
PhoneNlpDataset.__init__(self, size: int  =  1)
PhoneNlpDataset.__len__(self)
PhoneNlpDataset.get_phone_number(self, idx, islocal = False)
ImageDataset.__init__(self, path: Optional[Union[str, pathlib.Path]]  =  None, get_image_fn = None, meta = None, image_suffix = "*.png", **kwargs, )
ImageDataset.__len__(self)
ImageDataset.get_image_only(self, idx: int)
ImageDataset.get_sample(self, idx: int)
ImageDataset.get_label_list(self, label: Any)
ImageDataset.read_image(self, filepath_or_buffer: Union[str, io.BytesIO])
ImageDataset.save(self, path: str, prefix: str  =  "img", suffix: str  =  "png", nrows: int  =  -1)


utilmy/templates/templist/pypi_package/mygenerator/__init__.py


utilmy/templates/templist/pypi_package/mygenerator/util_image.py
-------------------------functions----------------------
padding_generate(paddings_number: int  =  1, min_padding: int  =  1, max_padding: int  =  1)
image_merge(image_list, n_dim, padding_size, max_height, total_width)
image_remove_extra_padding(img, inverse = False, removedot = True)
image_resize(image, width = None, height = None, inter = cv2.INTER_AREA)
image_read(filepath_or_buffer: Union[str, io.BytesIO])



utilmy/templates/templist/pypi_package/mygenerator/pipeline.py
-------------------------functions----------------------
run_generate_numbers_sequence(sequence: str, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, ### image_widthoutput_path: str  =  "./", config_file: str  =  "config/config.yaml", )
run_generate_phone_numbers(num_images: int  =  10, min_spacing: int  =  1, max_spacing: int  =  10, image_width: int  =  280, output_path: str  =  "./", config_file: str  =  "config/config.yaml", )



utilmy/templates/templist/pypi_package/mygenerator/utils.py
-------------------------functions----------------------
log(*s)
log2(*s)
logw(*s)
loge(*s)
logger_setup()
config_load(config_path: Optional[Union[str, pathlib.Path]]  =  None)
dataset_donwload(url, path_target)
dataset_get_path(cfg: dict)
os_extract_archive(file_path, path = ".", archive_format = "auto")
to_file(s, filep)



utilmy/templates/templist/pypi_package/mygenerator/transform.py
-------------------------methods----------------------
ImageTransform.__init__(self)
ImageTransform.transform(self, ds: dataset.ImageDataset)
ImageTransform.fit(self, ds: dataset.ImageDataset)
ImageTransform.fit_transform(self, ds: dataset.ImageDataset)
CharToImages.__init__(self, font: dataset.ImageDataset)
CharToImages.transform(self, ds: dataset.NlpDataset)
CharToImages.fit(self, ds: dataset.NlpDataset)
CharToImages.fit_transform(self, ds: dataset.NlpDataset)
RemoveWhitePadding.transform(self, ds: dataset.ImageDataset)
RemoveWhitePadding.transform_sample(self, image: np.ndarray)
CombineImagesHorizontally.__init__(self, padding_range: Tuple[int, int], combined_width: int)
CombineImagesHorizontally.transform(self, ds: dataset.ImageDataset)
CombineImagesHorizontally.transform_sample(self, image_list: List[np.ndarray], 1, 1), combined_width = 10, min_image_width = 2, validate = True, )
ScaleImage.__init__(self, width: Optional[int]  =  None, height: Optional[int]  =  None, inter = cv2.INTER_AREA)
ScaleImage.transform(self, ds: dataset.ImageDataset)
ScaleImage.transform_sample(self, image, width = None, height = None, inter = cv2.INTER_AREA)
TextToImage.__init__(self, font_dir: Union[str, pathlib.Path], spacing_range: Tuple[int, int], image_width: int)
TextToImage.transform(self, ds: dataset.NlpDataset)
TextToImage.fit(self, ds: dataset.NlpDataset)
TextToImage.fit_transform(self, ds: dataset.NlpDataset)


utilmy/templates/templist/pypi_package/tests/test_validate.py
-------------------------functions----------------------
test_image_padding_get()



utilmy/templates/templist/pypi_package/tests/test_transform.py
-------------------------functions----------------------
test_chars_to_images_transform()
test_combine_images_horizontally_transform()
test_scale_image_transform()
create_font_files(font_dir)
test_text_to_image_transform(tmp_path)



utilmy/templates/templist/pypi_package/tests/test_dataset.py
-------------------------functions----------------------
test_image_dataset_get_label_list()
test_image_dataset_len()
test_image_dataset_get_sampe()
test_image_dataset_get_image_only()
test_nlp_dataset_len()



utilmy/templates/templist/pypi_package/tests/test_util_image.py
-------------------------functions----------------------
create_blank_image(width, height, rgb_color = (0, 0, 0)
test_image_merge()
test_image_remove_extra_padding()
test_image_resize()
test_image_read(tmp_path)



utilmy/templates/templist/pypi_package/tests/__init__.py


utilmy/templates/templist/pypi_package/tests/test_import.py
-------------------------functions----------------------
test_import()



utilmy/templates/templist/pypi_package/tests/test_common.py


utilmy/templates/templist/pypi_package/tests/conftest.py


utilmy/templates/templist/pypi_package/tests/test_pipeline.py
-------------------------functions----------------------
test_generate_phone_numbers(tmp_path)



utilmy/zarchive/storage/aapackage_gen/34/util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/34/global01.py


utilmy/zarchive/storage/aapackage_gen/34/Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/old/util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/old/utils34.py
-------------------------functions----------------------
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
comoment(xx, yy, nsample, kx, ky)
acf(data)
unique_rows(a)
remove_zeros(vv, axis1 = 1)
sort_array(vv)
save_topanda(vv, filenameh5)
load_frompanda(filenameh5)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
parsePDF(url)



utilmy/zarchive/storage/aapackage_gen/old/util27.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/old/Working Copy of util34.py
-------------------------functions----------------------
getmodule_doc(module1, fileout = '')



utilmy/zarchive/storage/aapackage_gen/old/utils27.py
-------------------------functions----------------------
numexpr_vect_calc(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
numexpr_topanda(filename, i0 = 0, imax = 1000, expr, fileout='E = 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5')
convertcsv_topanda(filein1, filename, tablen = 'data')
getpanda_tonumpy(filename, nsize, tablen = 'data')
getrandom_tonumpy(filename, nbdim, nbsample, tablen = 'data')
comoment(xx, yy, nsample, kx, ky)
acf(data)
unique_rows(a)
remove_zeros(vv, axis1 = 1)
sort_array(vv)
save_topanda(vv, filenameh5)
load_frompanda(filenameh5)
plotsave(xx, yy, title1 = "")
plotshow(xx, yy, title1 = "")
parsePDF(url)

