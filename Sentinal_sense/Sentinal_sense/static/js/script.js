document.addEventListener('DOMContentLoaded', function() {
  const loginBtn = document.getElementById('loginBtn');
  const logoutBtn = document.getElementById('logoutBtn');
  const getInBtn = document.getElementById('getInBtn');

  loginBtn.addEventListener('click', function() {
    alert('You are now logged in.');
    loginBtn.style.display = 'none';
    logoutBtn.style.display = 'block';
  });

  logoutBtn.addEventListener('click', function() {
    alert('You are now logged out.');
    logoutBtn.style.display = 'none';
    loginBtn.style.display = 'block';
  });

  getInBtn.addEventListener('click', function() {
    alert('Welcome to our website!');
  });
});
