from django.shortcuts import render, HttpResponse, redirect
from firstApp import models
# Create your views here.


# 注册
def register(req):
    if req.method == 'GET':
        return render(req, 'register.html')
    if req.method == 'POST':
        username = req.POST.get('username')
        sex = req.POST.get('sex')
        phone = req.POST.get('phone')
        email = req.POST.get('email')
        password = req.POST.get('password')
        re_password = req.POST.get('re_password')
        if username and password and re_password:
            if password == re_password:
                user = models.UserInfor.objects.filter(username=username).first()
                if user:
                    return HttpResponse('用户已存在')
                else:
                    models.UserInfor.objects.create(
                        username=username,
                        password=password,
                        sex=sex,
                        phone=phone,
                        email=email
                    ).save()
                    return redirect('/login/')
            else:
                return HttpResponse('两次密码不一致')

        else:
            return HttpResponse('姓名，密码不能有空！')


# 登录
def login(req):
    if req.method == 'GET':
        return render(req, 'login.html')
    if req.method == 'POST':
        username = req.POST.get('username')
        password = req.POST.get('password')
        user = models.UserInfor.objects.filter(username=username, password=password).first()
        if user:
            return render(req, "index.html")
        else:
            return HttpResponse('用户名或密码错误！')


def index(req):
    if req.method == 'GET':
        return render(req, 'index.html')

# 查询
def query(req):
    if req.method == 'GET':
        bookname = req.GET.get('bookname')
        author = req.GET.get('author')
        book_list = models.BookInfor.objects.all()
        if not bookname and not author:
            book_list = models.BookInfor.objects.all()
        if bookname and author:
            book_list = models.BookInfor.objects.filter(bookname=bookname, author=author)
        if bookname and not author:
            book_list = models.BookInfor.objects.filter(bookname=bookname)
        if not bookname and author:
            book_list = models.BookInfor.objects.filter(author=author)
        return render(req, "query.html", {"book_list": book_list})

    return render(req, "query.html")


# 借阅
def borrow(req):
    if req.method == 'GET':
        book_list = models.BookInfor.objects.all()
        return render(req, "borrow.html", {"book_list": book_list})
    if req.method == 'POST':
        bookname = req.POST.get('bookname')
        if bookname:
            book = models.BookInfor.objects.filter(bookname=bookname).first()
            edit_id = book.id
            booknum = book.booknum
            if booknum == 0:
                return HttpResponse("图书库存为零")
            else:
                models.BookInfor.objects.filter(id=edit_id).update(booknum=booknum-1)
                return HttpResponse("借书成功")

    return render(req, "borrow.html")


# 还书
def give_back(req):
    if req.method == 'GET':
        return render(req, "give_back.html")
    if req.method == 'POST':
        bookname = req.POST.get('bookname')
        if bookname:
            book = models.BookInfor.objects.filter(bookname=bookname).first()
            edit_id = book.id
            booknum = book.booknum
            models.BookInfor.objects.filter(id=edit_id).update(booknum=booknum+1)
            return HttpResponse("还书成功")

    return render(req, "give_back.html")


# 添加图书
def bookadd(req):
    if req.method == 'GET':
        book_list = models.BookInfor.objects.all()
        return render(req, "bookadd.html", {"book_list": book_list})
    if req.method == "POST":
        name = req.POST.get("bookname", None)
        author = req.POST.get("author", None)
        isbn = req.POST.get("ISBN", None)
        pub = req.POST.get("publisher", None)
        number = req.POST.get("booknum", None)
        add = req.POST.get("address", None)
        # ---------表中插入数据方式一
        info = {"bookname": name, "author": author, "ISBN": isbn, "publisher": pub, "booknum": number, "address": add}
        '''
        # ---------表中插入数据方式二
        models.BookInfor.objects.create(
            bookname=name,
            ISBN=isbn,
            publisher=pub,
            booknum=number,
            address=add
        )
        '''
        models.BookInfor.objects.create(**info)
        info_list = models.BookInfor.objects.all()
        return render(req, "bookadd.html", {"info_list": info_list})

    return render(req, "bookadd.html")


# 删除图书
def delbook(req):
    if req.method == 'GET':
        book_list = models.BookInfor.objects.all()
        return render(req, "delbook.html", {"book_list": book_list})
    if req.method == 'POST':
        edit_id = req.POST.get('id')
        if edit_id:
            book = models.BookInfor.objects.filter(id=edit_id).first()
            if book:
                # 删除数据
                models.BookInfor.objects.filter(id=edit_id).delete()
                return HttpResponse("删除成功！")
            else:
                return HttpResponse("记录不存在！")
        else:
            return HttpResponse("编号不能为空！")

    return render(req, "delbook.html")


# 修改图书信息
def modbook(req):
    if req.method == 'GET':
        book_list = models.BookInfor.objects.all()
        return render(req, "modbook.html", {"book_list": book_list})
    if req.method == 'POST':
        edit_id = req.POST.get('id')
        bookname = req.POST.get('bookname')
        author = req.POST.get('author')
        ISBN = req.POST.get('ISBN')
        booknum = req.POST.get('booknum')
        publisher = req.POST.get('publisher')
        address = req.POST.get('address')
        if edit_id:
            book = models.BookInfor.objects.filter(id=edit_id).first()
            if book:
                # 修改数据
                models.BookInfor.objects.filter(id=edit_id).update(
                    bookname=bookname,
                    author=author,
                    ISBN=ISBN,
                    booknum=booknum,
                    publisher=publisher,
                    address=address
                )
                return HttpResponse("修改成功！")
            else:
                return HttpResponse("记录不存在！")
        else:
            return HttpResponse("编号不能为空！")
    return render(req, "modbook.html")
