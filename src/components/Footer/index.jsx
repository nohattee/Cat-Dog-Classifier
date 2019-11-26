import React from "react";
import styles from "./footer.module.css";

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className="container-fluid">
        <div className="row justify-content-left">
          <div className="col col-md">
            <p className={styles.description}>
              Nguyen Huu Tuan - 51704120<br></br>
              Ta Thanh Hung - 51704048
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
